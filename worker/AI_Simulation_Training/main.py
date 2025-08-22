import os
import json
import asyncio
import signal
from typing import Any, Dict

import aio_pika
from aio_pika import IncomingMessage

# Use relative imports to access modules within the 'worker' package
from .. import rabbitmq
from .ai import SalesPersonaAI

# This worker might need its own embedding model instance.
# For now, we assume it can be loaded on demand or is passed in the payload.
# A better approach would be a shared singleton.
EMBEDDING_MODEL = None

async def simulation_run(payload: dict) -> dict:
    """
    This is the core logic for the AI Simulation Training worker.
    It uses the SalesPersonaAI to generate a response.
    """
    global EMBEDDING_MODEL
    # Lazy load the embedding model once
    if EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            print("Embedding model loaded for simulation worker.")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            return {"error": "Embedding model could not be loaded."}

    # Extract required data from payload
    seller_msg = payload.get("prompt") # Assuming 'prompt' is the seller's message
    if not seller_msg:
        return {"error": "Seller message (prompt) is required."}

    # Reconstruct the AI state from the payload
    # The API service is responsible for managing and passing the session state
    ai_state = payload.get("state")
    if not ai_state:
        # If no state, create a new session (though this should ideally be handled by the API)
        ai = SalesPersonaAI(embedding_model=EMBEDDING_MODEL)
    else:
        ai = SalesPersonaAI.from_dict(ai_state, embedding_model=EMBEDDING_MODEL)

    # The stream_response method returns a generator.
    # We will consume it fully to get the complete response.
    response_chunks = []
    for chunk in ai.stream_response(seller_msg):
        response_chunks.append(chunk)

    full_response = "".join(response_chunks)

    # The worker should return the AI's response and the updated state
    return {
        "response": full_response,
        "state": ai.to_dict()
    }


# RabbitMQ consumer boilerplate
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
            await rabbitmq.publish_result(channel, "task.failed", {"error": f"invalid_json: {e}", "raw": body_text})
            return

        rk_in = message.routing_key or "unknown"
        job_id = payload.get("job_id")

        try:
            result = await simulation_run(payload)
            await rabbitmq.publish_result(channel, "task.succeeded", {"job_id": job_id, "routing_key": rk_in, "status": "succeeded", "result": result})
        except Exception as e:
            print(f"Error processing message {job_id}: {e}")
            await rabbitmq.publish_result(channel, "task.failed", {"job_id": job_id, "routing_key": rk_in, "status": "failed", "error": str(e)})
            return

async def main() -> None:
    """
    Main function to connect to RabbitMQ and start consuming messages
    specifically for the 'sim' queue.
    """
    print("Starting AI_Simulation_Training worker...")
    conn, ch, qs = await rabbitmq.connect_robust()

    sim_queue = qs.get("sim")
    if not sim_queue:
        print("Error: 'sim' queue not found.")
        await ch.close()
        await conn.close()
        return

    await sim_queue.consume(lambda m: handle_message(ch, m), no_ack=False)
    print("Worker is consuming messages from 'q.sim'. Waiting for jobs...")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()

    print("Shutting down AI_Simulation_Training worker.")
    await ch.close()
    await conn.close()

if __name__ == "__main__":
    # Note: Ensure sentence-transformers is in requirements.txt
    asyncio.run(main())
