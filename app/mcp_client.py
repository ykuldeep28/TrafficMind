import asyncio
import json
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]

class MCPToolClient:
    def __init__(self):
        self._ctx = None
        self._session = None
        self._lock = asyncio.Lock()

    async def start(self):
        if self._session:
            return
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        self._ctx = stdio_client(params)
        read, write = await self._ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

    async def call(self, name: str, arguments: dict):
        await self.start()
        async with self._lock:
            result = await self._session.call_tool(name, arguments)
            values = []
            for content in result.content:
                text = getattr(content, "text", None)
                if text:
                    try:
                        values.append(json.loads(text))
                    except json.JSONDecodeError:
                        values.append(text)
            return values[0] if len(values) == 1 else values

    async def close(self):
        # MCP/AnyIO contexts are created and closed in the same FastAPI
        # lifecycle task. This explicit shutdown keeps the v1 SDK happy.
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._ctx:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None

mcp_client = MCPToolClient()
