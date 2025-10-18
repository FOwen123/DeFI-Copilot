from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
from dataclasses import dataclass
from dotenv import load_dotenv
import time
import json
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("defillama")

BASE_URL = "https://api.llama.fi"

# HTTP client
client = httpx.AsyncClient(
    base_url=BASE_URL,
    headers={
        "accept": "application/json"
    },
    timeout=30.0
)

# Simple in-memory cache
_cache: Dict[str, tuple[Any, float]] = {}
CACHE_TTL = 60 * 60 # 1 hour

async def make_request(
    method: str, 
    endpoint: str, 
    params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> Any:
    """Make a request to the DefiLlama API with caching."""
    cache_key = f"{method}:{endpoint}:{json.dumps(params, sort_keys=True)}"
    
    if use_cache and cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            return cached_data
    
    try:
        response = await client.request(method, endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        
        _cache[cache_key] = (data, time.time())
        
        return data
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
            "success": False
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

# ============================================================================
# TVL ENDPOINTS
# ============================================================================

@mcp.tool()
async def get_protocols() -> str:
    """GET /protocols
    
    List all DeFi protocols along with their TVL and other metrics.
    Returns comprehensive data about all protocols tracked by DefiLlama.
    """
    result = await make_request('GET', '/protocols')
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_protocol(protocol: str) -> str:
    """GET /protocol/{protocol}
    
    Get detailed information about a specific protocol including current TVL,
    chain distribution, historical data, and metadata.
    
    Parameters:
        protocol: protocol slug (e.g., 'aave', 'uniswap', 'compound')
    
    Example: get_protocol('aave')
    """
    result = await make_request('GET', f'/protocol/{protocol}')
    return result

@mcp.tool()
async def get_tvl(protocol: str) -> str:
    """GET /tvl/{protocol}
    
    Get simplified current TVL for a protocol.
    
    Parameters:
        protocol: protocol slug (e.g., 'aave', 'uniswap')
    
    Returns just the TVL number for quick queries.
    """
    result = await make_request('GET', f'/tvl/{protocol}')
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_chains() -> str:
    """GET /v2/chains
    
    Get current TVL of all chains.
    Returns TVL for each blockchain with detailed breakdowns.
    """
    result = await make_request('GET', '/v2/chains')
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_historical_chain_tvl(chain: str) -> str:
    """GET /v2/historicalChainTvl/{chain}
    
    Get historical TVL data for a specific chain.
    
    Parameters:
        chain: chain name (e.g., 'Ethereum', 'Arbitrum', 'Polygon')
    """
    result = await make_request('GET', f'/v2/historicalChainTvl/{chain}')
    return json.dumps(result, indent=2)

@mcp.tool()
async def compare_protocols(protocols: str, metric: str = "tvl") -> str:
    """Custom tool to compare multiple protocols.
    
    Parameters:
        protocols: comma-separated protocol slugs (e.g., 'aave,compound,makerdao')
        metric: metric to compare - 'tvl', 'fees', or 'volume' (default: 'tvl')
    
    Returns comparison data for easy analysis.
    """
    protocol_list = [p.strip() for p in protocols.split(',')]
    results = {}
    
    for protocol in protocol_list:
        if metric == "tvl":
            data = await make_request('GET', f'/protocol/{protocol}')
        elif metric == "fees":
            data = await make_request('GET', f'/summary/fees/{protocol}')
        elif metric == "volume":
            data = await make_request('GET', f'/summary/dexs/{protocol}')
        else:
            data = {"error": f"Unknown metric: {metric}"}
        
        results[protocol] = data
    
    return json.dumps(results, indent=2)

# ============================================================================
# COINS ENDPOINTS
# ============================================================================

if __name__ == "__main__":
    
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*", "chrome-extension://*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    print(f"🚀 DeFi Llama MCP Server starting...")

    app.mount('/', mcp.sse_app())
    uvicorn.run(app, host='0.0.0.0', port=8000)

    