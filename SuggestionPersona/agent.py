from typing import Dict, Any, Optional, Tuple
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from jupyterlab_chat.models import Message, NewMessage
from jupyter_ydoc.ynotebook import YNotebook
import random
import numpy as np
import difflib
from langgraph.store.memory import InMemoryStore


import json
import asyncio
from typing_extensions import TypedDict
from jupyter_server_ai_tools import run_tools
from time import time
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


memory = MemorySaver()
in_memory_store = InMemoryStore()




def convert_mcp_to_openai(tools: list[dict]) -> list[dict]:
    """Convert a list of MCP-style tools to OpenAI-compatible function specs."""
    openai_tools = []

    for tool in tools:
        name = tool["name"]
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema", {})

        openai_tools.append(
            {
                "name": name,
                "description": description,
                "parameters": input_schema,
            }
        )

    return openai_tools


make_suggestions_metadata = {
    "name": "make_suggestion",
    "description": "Streams a suggestion message to the user."
    "in the notebook.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "suggestion_text": {"type": "string", "description": "The suggestion that the user will see."},
        },
        "required": ["suggestion_text"],
    },
}

class State(TypedDict):
    messages: list


async def create_langgraph_agent(
    extension_manager,
    logger,
    ychat,
    tools,
    notebook: YNotebook,
    self_id
):
    
    async def make_suggestion(suggestion_text: str) -> str:
            """Streams a suggestion message to the user. """
            stream_msg_id = ychat.add_message(NewMessage(body="", sender=self_id))
            current_text = ""

            for char in suggestion_text:
                await asyncio.sleep(0.01)
                current_text += char
                ychat.update_message(
                    Message(
                        id=stream_msg_id,
                        body=current_text,
                        time=time(),
                        sender=self_id,
                        raw_time=False,
                    ),
                    append=False,
                )
            return "Typing stream completed."
    raw_tools = tools
    logger.info(f"TOOL GROUPS: {raw_tools}")
    tool_groups = {t["metadata"]["name"]: t for t in raw_tools}
    # for now, to add imports the agent only needs read_notebook and write_to_cell
    tools = [
    t["metadata"]
    for t in raw_tools
    if t["metadata"]["name"] in ("read_notebook")
    ]

    tools.append(make_suggestions_metadata)
   


    logger.info(f"TOOLS: {tools}")

    llm = ChatOpenAI(api_key=api_key, model="gpt-4", temperature=0, streaming=True)
    openai_functions = convert_mcp_to_openai(tools)
    model = llm.bind_tools(tools=openai_functions)

    def parse_openai_tool_call(call: Dict) -> Tuple[str, Dict]:
        """
        Parses an OpenAI-style function call object and injects live objects like
        ynotebook or scheduler into the tool arguments based on the tool name.

        Returns:
            A tuple of (tool_name, arguments_dict)

        ---THIS IS A HACK TO GET AROUND PASSING AROUND LIVE INSTANCES OF THE NOTEBOOK---
        """
        fn = call.get("function", {})
        name = fn.get("name")
        arguments = fn.get("arguments", "{}")

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        # Inject the notebook for non-git tools
        if not name.startswith("git_"):
            logger.info("📎 Injecting ynotebook into tool arguments")
            arguments["ynotebook"] = notebook

        return name, arguments

    async def agent(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        stream_message_id = None
        full_chunk: Optional[AIMessageChunk] = None

        async for chunk in model.astream(messages):
            if full_chunk is None:
                full_chunk = chunk
            else:
                full_chunk += chunk

            content = getattr(chunk, "content", "")
            """
            if content:
                if stream_message_id is None:
                    stream_message_id = ychat.add_message(
                        NewMessage(body=content, sender=self_id)
                    )
                else:
                    ychat.update_message(
                        Message(
                            id=stream_message_id,
                            body=content,
                            time=time(),
                            sender=self_id,
                            raw_time=False,
                        ),
                        append=True,
                    )
            """
        full_message = AIMessage(
            content=full_chunk.content if full_chunk else "",
            additional_kwargs=full_chunk.additional_kwargs if full_chunk else {},
            response_metadata=full_chunk.response_metadata if full_chunk else {},
        )

        logger.info("✅ full_message: %s", full_message)
        return {"messages": messages + [full_message]}

    def should_continue(state):
        last_message = state["messages"][-1]

        # If the assistant has more tools to call
        if (
            isinstance(last_message, AIMessage)
            and "tool_calls" in last_message.additional_kwargs
        ):
            return "continue"

        # If we just handled a tool, go back to the agent
        if isinstance(last_message, ToolMessage):
            return "continue"

        # Otherwise, we're done
        return "end"

    async def call_tool(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        last_msg = messages[-1]
        tool_calls = last_msg.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) > 1:
            logger.warning("⚠️ Multiple tool calls detected. Only executing the first.")
            tool_calls = [tool_calls[0]]

        results = []

        for call in tool_calls:
            tool_name = call["function"]["name"]

            """# ✅ Stream "calling tool" message
            calling_msg = f"🔧 Calling {tool_name}...\n"
            stream_msg_id = ychat.add_message(NewMessage(body="", sender=self_id))

            for char in calling_msg:
                await asyncio.sleep(0.01)
                ychat.update_message(
                    Message(
                        id=stream_msg_id,
                        body=char,
                        time=time(),
                        sender=self_id,
                        raw_time=False,
                    ),
                    append=True,
            """

            if tool_name == "make_suggestion":
                parsed_name, args = parse_openai_tool_call(call)

                suggestion_text = args.get("suggestion_text")
                result = await make_suggestion(suggestion_text=suggestion_text)


            else: 
                # Run all other tools using the jupyter_server_ai_tools extension
                result = await run_tools(
                    extension_manager,
                    [call],
                    parse_fn=parse_openai_tool_call,
                )
            logger.info(f"TOOL RESULTS: {result}")
            tool_result = result

            # If the result is a coroutine, await it
            if asyncio.iscoroutine(tool_result):
                logger.warning(
                    "⚠️ Tool returned a coroutine — awaiting it before serialization."
                )
                tool_result = await tool_result

            tool_output = {"result": str(tool_result)}

            results.append((call, tool_output))
        # Format all results as ToolMessages

        tool_messages = [
            ToolMessage(
                name=call["function"]["name"],
                tool_call_id=call["id"],
                content=json.dumps(result_dict),
            )
            for call, result_dict in results
        ]

        logger.info(f"TOOL MESSAGES: {tool_messages}")

        return {"messages": state["messages"] + tool_messages}

    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.add_edge(START, "agent")
    workflow.add_node("call_tool", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "call_tool", "end": END}
    )
    workflow.add_edge("call_tool", "agent")

    compiled = workflow.compile(checkpointer=memory)
    return compiled


async def run_langgraph_agent(logger, agent, previous_suggestions):
    # 1) System prompt
    base_prompt = f"""
    You are a function-calling assistant operating inside a JupyterLab environment.
    Your job is to read the entire notebook and make suggestions. These should not be nit-picky suggestions on
    grammar or syntax. Rather, you should look at the notebook and suggest interesting or novel ways
    to explore and enhance what the user has already typed into the notebook.
    For example, you could suggest the code to make a graph. 
    If you see an error, and you know how to fix it please also suggest a fix for that via code. 
    Only suggest 2 or 3 things at a time. If you want to suggest code, suggest it in a way that it's easy to copy and paste for the user. 
    WHEN YOU HAVE A SUGGESTION - YOU MUST CALL make_suggestion OR ELSE THE USER WILL NOT SEE IT YOU MUST MAKE THAT TOOL CALL. 
    Please start by calling read_notebook. 
    I will list your previous suggestions. DO NOT repeat suggestions. If the user hasn't implemented the previous suggestions, 
    DO NOT make them again. Only make new suggestions, if any. It's OKAY to not have any new suggestions. 
    Here are your previous suggestions {previous_suggestions}. Please do not duplicate these suggestions. 
    """

    # Log existing history
    logger.info(f"HISTORY: {previous_suggestions}")

    messages = [{"role": "system", "content": base_prompt}]
    
    config = {"configurable": {"thread_id": "thread-1"}}
    logger.info(f"MESSAGES: {messages}")
    messages = await agent.ainvoke({"messages": messages}, config=config)
    return messages


    


async def create_supervisor_agent(logger, stream, tools):
    llm = ChatOpenAI(temperature=0, streaming=True).bind_tools(tools=tools)

    async def agent(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]

        response = await llm.ainvoke(messages)
        await stream(response.content or "")

        return {"messages": messages + [response]}

    def should_continue(state):
        last_message = state["messages"][-1]
        if (
            isinstance(last_message, AIMessage)
            and "tool_calls" in last_message.additional_kwargs
        ):
            return "continue"
        if isinstance(last_message, ToolMessage):
            return "continue"
        return "end"

    async def call_tool(state: Dict[str, Any]) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = last_msg.additional_kwargs.get("tool_calls", [])
        results = []

        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            logger.info(f"Calling tool: {name} with args: {args}")

            # Look up tool by name
            for tool in tools:
                if tool.name == name:
                    result = await tool.ainvoke(args)
                    await stream(f"🔧 {name} executed: {result}")
                    tool_message = ToolMessage(
                        tool_call_id=call["id"],
                        name=name,
                        content=json.dumps({"result": result}),
                    )
                    results.append(tool_message)
                    break
            else:
                logger.warning(f"No tool found for {name}")

        return {"messages": state["messages"] + results}

    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.add_node("call_tool", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "call_tool", "end": END}
    )
    workflow.add_edge("call_tool", "agent")

    compiled = workflow.compile(
        checkpointer=memory,
        store=in_memory_store
        
    )
    return compiled



async def run_supervisor_agent(logger, compiled, user_message, message_history): 
    system_prompt = """
        You are a helpful assistant that can start or stop a collaborative editng session with an agent.
        If the user tells you "my name is X", you should remember it and be able to recall it later in the session.
        You are operating in a collaborative JupyterLab session.
        """
    
    
    if message_history != None: 
        logger.info(f"MESSAGE HISTORY: {message_history}")
        prev = message_history["messages"]
        logger.info(f"PREV: {prev}")
        prev.append({"role": "system", "content": system_prompt})
        prev.append({"role": "user", "content": user_message})
        messages = prev.copy()
    else: 
        messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}]


    # ✅ Append the new user message
    config = {
        "configurable": {
            "thread_id": f"notebook-session"
        }
    }


    logger.info(f"MESSAGES: {messages}")
    #logger.info(f"CHECKPOINTS: {checkpoints}")
    result = await compiled.ainvoke(
        {"messages": messages},
        config=config
    )
    return result

  
