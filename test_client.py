"""Client for the MCP server using Server-Sent Events (SSE)."""

import asyncio

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    """
    Main function to demonstrate MCP client functionality.

    Establishes an SSE connection to the server, initializes a session,
    and demonstrates basic operations like sending pings, listing tools,
    and calling a weather tool.
    """
    async with sse_client(url="http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.send_ping()
            tools = await session.list_tools()

            for tool in tools.tools:
                print("Name:", tool.name)
                print("Description:", tool.description)
            print()

            chains = await session.call_tool(
                name="get_chains", arguments={}
            )
            print("Tool Call")
            print(chains)

asyncio.run(main())