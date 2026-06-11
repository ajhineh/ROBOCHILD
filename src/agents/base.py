import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class AgentMessage:
    """Standard message container for agent-to-agent communication"""
    def __init__(self, sender: str, recipient: str, message_type: str, data: Dict[str, Any]):
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type  # e.g., 'task', 'result', 'error', 'event'
        self.data = data

class BaseAgent(ABC):
    """Abstract base class representing an Agent"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.logger = logging.getLogger(name)
        self.inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.is_running = False

    async def send_message(self, recipient_agent: 'BaseAgent', message_type: str, data: Dict[str, Any]):
        """Asynchronously send a message to another agent's inbox"""
        msg = AgentMessage(sender=self.name, recipient=recipient_agent.name, message_type=message_type, data=data)
        await recipient_agent.inbox.put(msg)

    async def start(self):
        """Starts the agent main loop"""
        self.is_running = True
        self.logger.info(f"Agent {self.name} ({self.role}) started.")
        asyncio.create_task(self._main_loop())

    async def stop(self):
        """Stops the agent main loop"""
        self.is_running = False
        self.logger.info(f"Agent {self.name} stopped.")

    async def _main_loop(self):
        """Main loop that continuously polls the agent's inbox"""
        while self.is_running:
            try:
                # Wait for next message with a timeout to allow graceful shutdowns
                message = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                await self.handle_message(message)
                self.inbox.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error handling message: {e}")

    @abstractmethod
    async def handle_message(self, message: AgentMessage):
        """Handles incoming messages. Must be implemented by specialists."""
        pass
