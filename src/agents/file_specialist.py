import os
import aiofiles
import json
from datetime import datetime
from .base import BaseAgent, AgentMessage

class FileSpecialistAgent(BaseAgent):
    """
    File Specialist Agent: Safely handles file operations, log writing, and state persistance.
    """
    def __init__(self, name: str = "file_specialist", log_dir: str = "logs"):
        super().__init__(name, "Log writer & File persistence Specialist")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    async def handle_message(self, message: AgentMessage):
        if message.message_type == "log_trade":
            await self._log_to_file("trades_history.jsonl", message.data)
        elif message.message_type == "log_signal":
            await self._log_to_file("signals_history.jsonl", message.data)

    async def _log_to_file(self, filename: str, data: dict):
        filepath = os.path.join(self.log_dir, filename)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }
        try:
            async with aiofiles.open(filepath, mode="a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Failed writing log to {filename}: {e}")
