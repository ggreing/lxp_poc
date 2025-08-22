import os
import json
import asyncio
import signal
from typing import Any, Dict

import aio_pika
from aio_pika import IncomingMessage
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis

from .. import rabbitmq
from .ai import SalesPersonaAI

# --- Global State ---
EMBEDDING_MODEL = None
REDIS_CLIENT = None

def get_embedding_model():
    """Lazy-loads the embedding model."""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model for simulation worker...")
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded.")
    return EMBEDDING_MODEL

def get_redis_client():
    """Lazy-loads the redis client."""
    global REDIS_CLIENT
    if REDIS_CLIENT is None:
        print("Connecting to Redis...")
        redis_host = os.getenv("REDIS_HOST", "redis")
        REDIS_CLIENT = redis.from_url(f"redis://{redis_host}", encoding="utf-8", decode_responses=False)
        print("Redis client connected.")
    return REDIS_CLIENT

def RKEY_SESSION(session_id: str): return f"sim_session:{session_id}"

# --- Worker Logic ---

async def handle_start_session(channel: Any, payload: dict):
    """Handles the start_session task."""
    session_id = payload.get("session_id")
    persona = payload.get("persona")

    ai = SalesPersonaAI(
        persona=persona,
        session_id=session_id,
        user_id=payload.get("user_id"),
        embedding_model=get_embedding_model()
    )

    greeting = ai.generate_first_greeting()
    ai._append_history("AI", greeting)

    # Save initial state to Redis
    redis_client = get_redis_client()
    await redis_client.set(RKEY_SESSION(session_id), json.dumps(ai.to_dict()), ex=3600) # 1 hour expiry

    # Send the greeting to the chat stream
    await rabbitmq.publish_chat_response(channel, session_id, greeting, event="greeting")

async def handle_chat_message(channel: Any, payload: dict):
    """Handles a regular chat message."""
    session_id = payload.get("session_id")
    seller_msg = payload.get("seller_msg")

    if not seller_msg:
        return

    redis_client = get_redis_client()
    state_data = await redis_client.get(RKEY_SESSION(session_id))
    if not state_data:
        await rabbitmq.publish_chat_response(channel, session_id, "Error: Session not found or expired.", event="error")
        return

    ai_state = json.loads(state_data)
    ai = SalesPersonaAI.from_dict(ai_state, embedding_model=get_embedding_model())

    full_response = "".join(ai.stream_response(seller_msg))

    # Save updated state
    await redis_client.set(RKEY_SESSION(session_id), json.dumps(ai.to_dict()), ex=3600)

    # Publish the response to the chat stream
    await rabbitmq.publish_chat_response(channel, session_id, full_response, event="message")

# --- RabbitMQ Boilerplate ---

async def handle_message(channel: Any, message: IncomingMessage):
    async with message.process(requeue=False):
        try:
            payload = json.loads(message.body.decode("utf-8") or "{}")
            routing_key = message.routing_key or ""

            if routing_key.endswith(".start"):
                await handle_start_session(channel, payload)
            elif routing_key.endswith(".chat"):
                await handle_chat_message(channel, payload)
            else:
                print(f"Unknown task for routing key: {routing_key}")

        except Exception as e:
            print(f"Error processing message: {e}")

async def main() -> None:
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
    if REDIS_CLIENT:
        await REDIS_CLIENT.close()
    await ch.close()
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
