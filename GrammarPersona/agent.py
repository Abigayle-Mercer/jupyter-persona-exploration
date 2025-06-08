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


def human_typing_delay(
    base=0.01,
    std=0.03,
    min_delay=0.002,
    max_delay=0.2,
):
    delay = np.random.normal(loc=base, scale=std)
    return max(min_delay, min(delay, max_delay))


async def write_to_cell(
    ynotebook: YNotebook, index: int, content: str, stream: bool = True
) -> str:
    """
    Overwrite the source of a notebook cell at the given index with human-like typing.

    Parameters:
        ynotebook (YNotebook): The notebook to modify.
        index (int): The index of the cell to overwrite.
        content (str): The new content to write.
        stream (bool): Whether to simulate gradual updates (default: True).

    Returns:
        str: Success or error message.

    ---THIS IS TEMPERARY, MAY BE MERGED INTO JUPYTER-AI-TOOLS IF SUCCESSFUL---
    """
    try:
        ycell = ynotebook.get_cell(index)
        old = ycell["source"]
        new = content

        if not stream:
            ycell["source"] = new
            ynotebook.set_cell(index, ycell)
            return f"✅ Overwrote cell {index}."

        sm = difflib.SequenceMatcher(None, old, new)
        result = list(old)
        cursor = 0

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                cursor += i2 - i1

            elif tag == "delete":
                for offset in reversed(range(i2 - i1)):
                    del result[cursor + offset]
                    ycell["source"] = "".join(result)
                    ynotebook.set_cell(index, ycell)
                    await asyncio.sleep(human_typing_delay())

            elif tag == "insert":
                for i, c in enumerate(new[j1:j2]):
                    result.insert(cursor, c)
                    cursor += 1
                    ycell["source"] = "".join(result)
                    ynotebook.set_cell(index, ycell)

                    if c in {".", "!", "?", "\n"}:
                        await asyncio.sleep(human_typing_delay() + 0.1)
                    elif c == " " and random.random() < 0.2:
                        await asyncio.sleep(0.1)
                    elif random.random() < 0.02:
                        await asyncio.sleep(0.15)
                    else:
                        await asyncio.sleep(human_typing_delay())

            elif tag == "replace":
                for _ in range(i2 - i1):
                    result.pop(cursor)
                    ycell["source"] = "".join(result)
                    ynotebook.set_cell(index, ycell)
                    await asyncio.sleep(human_typing_delay())

                for i, c in enumerate(new[j1:j2]):
                    result.insert(cursor, c)
                    cursor += 1
                    ycell["source"] = "".join(result)
                    ynotebook.set_cell(index, ycell)

                    if c in {".", "!", "?", "\n"}:
                        await asyncio.sleep(human_typing_delay() + 0.1)
                    elif c == " " and random.random() < 0.2:
                        await asyncio.sleep(0.1)
                    elif random.random() < 0.08:
                        await asyncio.sleep(0.15)
                    else:
                        await asyncio.sleep(human_typing_delay())

        return f"✅ Updated cell {index}."
    except Exception as e:
        return f"❌ Error editing cell {index}: {str(e)}"


class State(TypedDict):
    messages: list


async def run_langgraph_agent(
    extension_manager,
    logger,
    ychat,
    tools,
    notebook: YNotebook,
    user_prompt: str,
    tone_prompt,
    self_id,
    get_active_cell,
    append_edited_cells,
    get_edited_cells,
):
    raw_tools = tools
    logger.info(f"TOOL GROUPS: {raw_tools}")
    tool_groups = {t["metadata"]["name"]: t for t in raw_tools}
    tools = [t["metadata"] for t in raw_tools]

    logger.info(f"TOOLS: {tools}")

    memory = MemorySaver()
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

            # ✅ Stream "calling tool" message
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
                )

            # if write_to_cell, call the custom one: THIS IS JUST FOR TESTING PURPOSES
            if tool_name == "write_to_cell":
                # Parse arguments from the tool call
                parsed_name, args = parse_openai_tool_call(call)

                index = args.get("index")
                content = args.get("content")
                stream = args.get("stream", True)

                if index is None or content is None:
                    logger.warning("⚠️ Missing required arguments for write_to_cell.")
                    tool_result = (
                        "❌ Missing 'index' or 'content' argument for write_to_cell."
                    )
                else:
                    result = await write_to_cell(
                        ynotebook=notebook, index=index, content=content, stream=stream
                    )
                    cell = notebook.get_cell(index)
                    cell_id = cell.get("id") or cell.get("metadata", {}).get("id")
                    if result.startswith("✅"):
                        append_edited_cells(cell_id)
            else:
                # Run all other tools using the jupyter_server_ai_tools extension
                result = await run_tools(
                    extension_manager,
                    [call],
                    parse_fn=parse_openai_tool_call,
                )
            logger.info(f"TOOL RESULTS: {result}")
            tool_result = result[0]

            # If the result is a coroutine, await it
            if asyncio.iscoroutine(tool_result):
                logger.warning(
                    "⚠️ Tool returned a coroutine — awaiting it before serialization."
                )
                tool_result = await tool_result

            tool_output = {"result": str(tool_result)}

            # If it's a notebook-mutating tool, capture post-edit state - makes sure agent agnet is up to date in long running workflows
            if tool_name in {"write_to_cell", "add_cell", "delete_cell"}:
                read_nb = tool_groups["read_notebook"]["callable"]

                try:
                    notebook_contents_str = await read_nb(notebook)
                    notebook_cells = json.loads(notebook_contents_str)
                except Exception as e:
                    logger.warning(f"❌ Failed to read or parse notebook contents: {e}")
                    notebook_cells = []

                try:
                    active_cell_id = get_active_cell(notebook)
                except Exception as e:
                    logger.warning(f"❌ Failed to get active cell ID: {e}")
                    active_cell_id = None

                edited_cells = get_edited_cells()

                tool_output["notebook_snapshot"] = {
                    "cells": notebook_cells,
                    "activeCellId": active_cell_id,
                    "editedCells": edited_cells,
                }

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

    system_prompt = f"""
    You are a function-calling assistant operating inside a JupyterLab environment.
    Your job is to read the entire notebook and if you find grammar mistakes in markdown cells, 
    and fix them using your available tools. Only operate on markdown cells please. 
    Along with fixing grammatical mistakes, please adapt markdown text to the following tone: {tone_prompt}
    You may only call one tool at a time. If you want to perform multiple actions, wait for confirmation and state each one step by step.
    Please focus on tool calls and not sending messages to the user. 
    Please start by calling read_notebook.
    
    """

    logger.info(f"PROMPT: {system_prompt}")
    logger.info(f"PROMPT: {user_prompt}")
    state = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }
    config = {"configurable": {"thread_id": "thread-1"}}

    await compiled.ainvoke(state, config=config)


async def run_supervisor_agent(logger, stream, user_message: str, tools):
    memory = MemorySaver()
    llm = ChatOpenAI(temperature=0, streaming=True).bind_tools(tools=tools)

    async def agent(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        stream_message_id = None
        full_chunk: Optional[AIMessageChunk] = None

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

    compiled = workflow.compile(checkpointer=memory)

    system_prompt = """
        You are a helpful assistant that can start or stop a collaborative grammar editing session.
        The user may describe their tone preferences for the way they want their markdown cells edited. 
        """
    state = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    }
    config = {"configurable": {"thread_id": "thread-1"}}
    logger.info("HERE")
    await compiled.ainvoke(state, config=config)
