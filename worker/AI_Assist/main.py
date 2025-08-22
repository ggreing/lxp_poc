import os
import json
import asyncio
import signal
from typing import Any, Dict

import aio_pika
from aio_pika import IncomingMessage
import httpx

# Since this file is in worker/AI_Assist, we need to adjust the import path
# to access the rabbitmq module in the parent 'worker' directory.
from .. import rabbitmq

# Logic from the original ai_assist.py
API_URL = os.getenv("API_URL", "http://api:8000")

async def assist_run(payload: dict) -> dict:
    """
    This is the core logic for the AI Assist worker.
    It calls the RAG API to get an answer.
    """
    prompt = payload.get("prompt") or ""
    vs_id = payload.get("vectorstore_id")

    if not vs_id or not prompt:
        return {"answer": "Vector store ID and prompt are required.", "evidence": []}

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Note: The original endpoint was '/sales/rag/query'. This might need to be
            # generalized in the future, but we'll stick to it for now.
            response = await client.post(
                f"{API_URL}/rag/query",
                json={
                    "prompt": prompt,
                    "vectorstore_id": vs_id,
                    "top_k": 3
                }
            )
            response.raise_for_status()
            result = response.json()
            result["sub_function"] = payload.get("sub_function") or "qa"
            return result
    except httpx.HTTPStatusError as e:
        error_message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        print(error_message)
        return {"answer": error_message, "evidence": []}
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return {"answer": f"An unexpected error occurred: {str(e)}", "evidence": []}

# RabbitMQ consumer logic adapted from the old monolithic worker main.py
async def handle_message(channel: Any, message: IncomingMessage):
    """
    Handles incoming messages from RabbitMQ, calls the worker logic,
    and publishes the result.
    """
    async with message.process(requeue=False):
        body_text = message.body.decode("utf-8") if message.body else "{}"
        try:
            payload = json.loads(body_text or "{}")
        except Exception as e:
            await rabbitmq.publish_result(channel, "task.failed", {
                "error": f"invalid_json: {e}", "raw": body_text
            })
            return

        rk_in = message.routing_key or "unknown"
        job_id = payload.get("job_id")

        try:
            # Directly call the worker's run function
            result = await assist_run(payload)
            await rabbitmq.publish_result(channel, "task.succeeded", {
                "job_id": job_id, "routing_key": rk_in, "status": "succeeded", "result": result
            })
        except Exception as e:
            print(f"Error processing message {job_id}: {e}")
            await rabbitmq.publish_result(channel, "task.failed", {
                "job_id": job_id, "routing_key": rk_in, "status": "failed", "error": str(e)
            })
            return

async def main() -> None:
    """
    Main function to connect to RabbitMQ and start consuming messages
    specifically for the 'assist' queue.
    """
    print("Starting AI_Assist worker...")
    conn, ch, qs = await rabbitmq.connect_robust()

    # Get the specific queue for the 'assist' worker
    assist_queue = qs.get("assist")
    if not assist_queue:
        print("Error: 'assist' queue not found. Make sure it's defined in rabbitmq.py.")
        await ch.close()
        await conn.close()
        return

    await assist_queue.consume(lambda m: handle_message(ch, m), no_ack=False)

    print("Worker is consuming messages from 'q.assist'. Waiting for jobs...")

    # Wait for termination signal
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()

    print("Shutting down AI_Assist worker.")
    await ch.close()
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
