import os
import json
import asyncio
import signal
from typing import Any, Dict, List
import pandas as pd

import aio_pika
from aio_pika import IncomingMessage

from .. import rabbitmq

# --- Course Recommendation Logic ---

def load_course_data():
    """
    Loads course data from the CSV file.
    In a real application, this might connect to a database.
    The CSV is assumed to be in the root of the repository.
    """
    # Robustly find the CSV file path
    # This assumes the worker is run from the repository root,
    # or the path is otherwise accessible.
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'course_data.csv')
    if not os.path.exists(csv_path):
        print(f"Error: course_data.csv not found at {csv_path}")
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading or parsing course_data.csv: {e}")
        return None

COURSE_DF = load_course_data()

async def recommend_courses(prompt: str) -> List[Dict[str, Any]]:
    """
    Recommends courses based on a user prompt.
    This is a simple keyword matching implementation.
    """
    if COURSE_DF is None:
        return [{"error": "Course data not available."}]

    if not prompt:
        return []

    # Simple keyword search in title, keywords, and objectives
    keywords = prompt.lower().split()

    # Create a boolean series for each keyword
    matches = [
        COURSE_DF['course_title'].str.lower().str.contains(kw) |
        COURSE_DF['keywords'].str.lower().str.contains(kw) |
        COURSE_DF['learning_objectives'].str.lower().str.contains(kw)
        for kw in keywords
    ]

    # Combine matches: a course matches if it contains any of the keywords
    combined_matches = pd.concat(matches, axis=1).any(axis=1)

    results = COURSE_DF[combined_matches]

    # Return top 3 results, converted to dictionary format
    return results.head(3).to_dict(orient='records')


# --- RabbitMQ Worker boilerplate ---

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
        prompt = payload.get("prompt")

        try:
            result = await recommend_courses(prompt)
            await rabbitmq.publish_result(channel, "task.succeeded", {"job_id": job_id, "routing_key": rk_in, "status": "succeeded", "result": result})
        except Exception as e:
            print(f"Error processing message {job_id}: {e}")
            await rabbitmq.publish_result(channel, "task.failed", {"job_id": job_id, "routing_key": rk_in, "status": "failed", "error": str(e)})
            return

async def main() -> None:
    """
    Main function to connect to RabbitMQ and start consuming messages
    specifically for the 'coach' queue.
    """
    print("Starting Galaxy_Coach worker...")
    conn, ch, qs = await rabbitmq.connect_robust()

    coach_queue = qs.get("coach")
    if not coach_queue:
        print("Error: 'coach' queue not found.")
        await ch.close()
        await conn.close()
        return

    await coach_queue.consume(lambda m: handle_message(ch, m), no_ack=False)

    print("Worker is consuming messages from 'q.coach'. Waiting for jobs...")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()

    print("Shutting down Galaxy_Coach worker.")
    await ch.close()
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
