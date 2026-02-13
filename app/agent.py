import json
import operator
from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
from langchain_community.tools import PolygonLastQuote, PolygonTickerNews, PolygonFinancials, PolygonAggregates
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_core.messages import BaseMessage, FunctionMessage, convert_to_messages
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph import StateGraph, END

import os
from app.tools import discounted_cash_flow, owner_earnings, roic, roe

# Load the environment variables
load_dotenv()

# Choose the LLM that will drive the agent
model_provider = os.getenv("MODEL_PROVIDER", "openai").lower()

if model_provider == "groq":
    model_name = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    model = ChatGroq(model=model_name, streaming=True)
else:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4-0125-preview")
    model = ChatOpenAI(model=model_name, streaming=True)

# Create the tools
polygon = PolygonAPIWrapper()
integration_tools = [
    PolygonLastQuote(api_wrapper=polygon),
    PolygonTickerNews(api_wrapper=polygon),
    PolygonFinancials(api_wrapper=polygon),
    PolygonAggregates(api_wrapper=polygon),
]

local_tools = [discounted_cash_flow, roe, roic, owner_earnings]
tools = integration_tools + local_tools

tools_by_name = {t.name: t for t in tools}

# Groq uses bind_tools instead of bind_functions for newer models, 
# but for compatibility with the existing code structure, we use bind_tools if available.
if hasattr(model, "bind_tools"):
    model = model.bind_tools(tools)
else:
    functions = [convert_to_openai_function(t) for t in tools]
    model = model.bind_functions(functions)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function/tool call, then we finish
    if "function_call" not in last_message.additional_kwargs and (not hasattr(last_message, "tool_calls") or not last_message.tool_calls):
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    # Convert messages to a robust format (list of tuples) to avoid Pydantic/type issues
    messages = []
    for m in state['messages']:
        if hasattr(m, "type"):
            role = m.type
            if role == "human": role = "user"
            elif role == "ai": role = "assistant"
            elif role == "function": role = "function"
            
            # For assistant messages with tool calls, we need to preserve them
            if role == "assistant" and hasattr(m, "tool_calls") and m.tool_calls:
                 messages.append(m)
            else:
                 messages.append((role, m.content))
        else:
            messages.append(m)
            
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state['messages']
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    
    # Handle OpenAI-style function calls and standard tool calls
    if "function_call" in last_message.additional_kwargs:
        tool_name = last_message.additional_kwargs["function_call"]["name"]
        tool_input = json.loads(last_message.additional_kwargs["function_call"]["arguments"])
    elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Take the first tool call
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
    else:
        raise ValueError("No tool call found in the last message.")

    # Invoke the tool directly
    tool = tools_by_name[tool_name]
    response = tool.invoke(tool_input)
    
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=tool_name)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # END is a special node marking that the graph should finish.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
workflow.add_edge('action', 'agent')

# Finally, we compile it!
agent = workflow.compile()
