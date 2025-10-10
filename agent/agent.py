from dotenv import load_dotenv
import operator
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()


# Define the agent state
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: dict  # For DeFi-specific context (wallet, chain, etc.)


# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True
)


def create_defi_agent():
    """Create and return the DeFi Copilot LangGraph agent."""
    
    # Define the system prompt for DeFi assistance
    system_prompt = """You are DeFi Copilot, an expert AI assistant specialized in decentralized finance (DeFi).

Your capabilities include:
- Analyzing DeFi protocols and smart contracts
- Explaining tokenomics and protocol mechanics
- Providing security assessments and risk analysis
- Helping users understand yield farming, staking, and liquidity provision
- Analyzing on-chain data and transactions
- Explaining blockchain concepts in simple terms

Always prioritize user safety and clearly communicate risks. Be concise but thorough."""

    def agent_node(state: AgentState):
        """Main agent reasoning node."""
        messages = state["messages"]
        
        # Add system prompt if this is the first message
        if len(messages) == 1 or not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        # Get context if available
        context = state.get("context", {})
        
        # Add context to the last message if present
        if context:
            last_msg = messages[-1]
            context_str = f"\n\nContext: {context}"
            if isinstance(last_msg, HumanMessage):
                messages[-1] = HumanMessage(content=last_msg.content + context_str)
        
        # Call the LLM
        response = llm.invoke(messages)
        
        return {"messages": [response]}

    def should_continue(state: AgentState):
        """Determine if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # For now, always end after agent responds
        # Later you can add tool calling logic here
        return "end"

    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "end": END,
            # Later: "continue": "tools" for tool usage
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    return app


# Create the agent instance
agent = create_defi_agent()


async def stream_agent_response(query: str, context: dict = None):
    """
    Stream responses from the agent.
    
    Args:
        query: User's question or command
        context: Optional DeFi context (wallet address, chain, page content, etc.)
    
    Yields:
        Chunks of the agent's response
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "context": context or {}
    }
    
    async for event in agent.astream(initial_state, stream_mode="values"):
        if "messages" in event:
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage):
                yield last_message.content


def invoke_agent(query: str, context: dict = None):
    """
    Synchronous invoke for the agent.
    
    Args:
        query: User's question or command
        context: Optional DeFi context
    
    Returns:
        The agent's response as a string
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "context": context or {}
    }
    
    result = agent.invoke(initial_state)
    return result["messages"][-1].content


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        # Test with a simple query
        print("Testing DeFi Copilot Agent...\n")
        
        query = "Explain what liquidity pools are in DeFi"
        
        print(f"Query: {query}\n")
        print("Response:")
        
        async for chunk in stream_agent_response(query):
            print(chunk, end="", flush=True)
        
        print("\n\nDone!")
    
    asyncio.run(test_agent())