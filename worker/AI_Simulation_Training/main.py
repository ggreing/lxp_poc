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

# --- Constants ---
RKEY_SESSION = lambda session_id: f"sim_session:{session_id}"

# --- Worker Class ---

class SimulationWorker:
    """
    A worker class that encapsulates all resources and logic for handling
    AI simulation tasks. This avoids global state for model and clients.
    """
    def __init__(self):
        self.embedding_model = None
        self.redis_client = None
        self.rabbitmq_conn = None
        self.rabbitmq_channel = None

    async def _initialize_resources(self):
        """Initializes the embedding model and Redis client."""
        print("Loading embedding model for simulation worker...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded.")

        print("Connecting to Redis...")
        redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_client = redis.from_url(f"redis://{redis_host}", encoding="utf-8", decode_responses=False)
        print("Redis client connected.")

    async def _handle_start_session(self, payload: dict):
        """Handles the start_session task."""
        session_id = payload.get("session_id")
        persona = payload.get("persona")

        ai = SalesPersonaAI(
            persona=persona,
            session_id=session_id,
            user_id=payload.get("user_id"),
            embedding_model=self.embedding_model
        )

        greeting = ai.generate_first_greeting()
        ai._append_history("AI", greeting)

        # Save initial state to Redis
        await self.redis_client.set(RKEY_SESSION(session_id), json.dumps(ai.to_dict()), ex=3600) # 1 hour expiry

        # Send the greeting to the chat stream
        await rabbitmq.publish_chat_response(self.rabbitmq_channel, session_id, greeting, event="greeting")

    async def _handle_chat_message(self, payload: dict):
        """Handles a regular chat message, streaming the response."""
        session_id = payload.get("session_id")
        seller_msg = payload.get("seller_msg")

        if not seller_msg:
            return

        state_data = await self.redis_client.get(RKEY_SESSION(session_id))
        if not state_data:
            await rabbitmq.publish_chat_response(self.rabbitmq_channel, session_id, "Error: Session not found or expired.", event="error")
            return

        ai_state = json.loads(state_data)
        ai = SalesPersonaAI.from_dict(ai_state, embedding_model=self.embedding_model)

        # Stream the response and publish chunks to RabbitMQ
        try:
            # stream_response is a generator that yields chunks of the response.
            # It also handles updating the AI's internal history.
            for chunk in ai.stream_response(seller_msg):
                await rabbitmq.publish_chat_response(
                    self.rabbitmq_channel,
                    session_id,
                    chunk,
                    event="message" # Client should append text for this event.
                )

            # After streaming is complete, save the final state of the AI object back to Redis.
            await self.redis_client.set(RKEY_SESSION(session_id), json.dumps(ai.to_dict()), ex=3600)

            # Signal the end of the stream to the client.
            await rabbitmq.publish_chat_response(self.rabbitmq_channel, session_id, "", event="message_end")

        except Exception as e:
            print(f"Error during chat streaming for session {session_id}: {e}")
            await rabbitmq.publish_chat_response(self.rabbitmq_channel, session_id, f"Error: An error occurred while generating the response.", event="error")

    async def _on_message(self, message: IncomingMessage):
        """Main callback for processing messages from RabbitMQ."""
        async with message.process(requeue=False):
            try:
                payload = json.loads(message.body.decode("utf-8") or "{}")
                routing_key = message.routing_key or ""

                if routing_key.endswith(".start"):
                    await self._handle_start_session(payload)
                elif routing_key.endswith(".chat"):
                    await self._handle_chat_message(payload)
                else:
                    print(f"Unknown task for routing key: {routing_key}")

            except Exception as e:
                print(f"Error processing message: {e}")

    async def run(self):
        """Connects to services, starts consuming, and handles graceful shutdown."""
        print("Starting AI_Simulation_Training worker...")
        await self._initialize_resources()

        self.rabbitmq_conn, self.rabbitmq_channel, qs = await rabbitmq.connect_robust()
        sim_queue = qs.get("sim")
        if not sim_queue:
            print("Error: 'sim' queue not found.")
            await self.shutdown()
            return

        await sim_queue.consume(self._on_message, no_ack=False)
        print("Worker is consuming messages from 'q.sim'. Waiting for jobs...")

        # Wait for termination signal
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        await stop_event.wait()

        print("Termination signal received.")
        await self.shutdown()

    async def shutdown(self):
        """Closes all connections gracefully."""
        print("Shutting down AI_Simulation_Training worker.")
        if self.redis_client:
            await self.redis_client.close()
            print("Redis client closed.")
        if self.rabbitmq_channel and not self.rabbitmq_channel.is_closed:
            await self.rabbitmq_channel.close()
            print("RabbitMQ channel closed.")
        if self.rabbitmq_conn and not self.rabbitmq_conn.is_closed:
            await self.rabbitmq_conn.close()
            print("RabbitMQ connection closed.")

async def main() -> None:
    worker = SimulationWorker()
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
