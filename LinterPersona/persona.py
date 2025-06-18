import asyncio
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from time import time
from jupyterlab_chat.models import Message, NewMessage
from jupyter_ydoc.ynotebook import YNotebook
from jupyter_server.base.call_context import CallContext
from typing_extensions import TypedDict
from jupyter_server_ai_tools import find_tools
from jupyter_ydoc.ynotebook import YNotebook
from .agent import run_langgraph_agent as external_run_langgraph_agent
from .agent import create_langgraph_agent as external_create_langgraph_agent
from langchain_core.tools import tool

from .agent import create_supervisor_agent, run_supervisor_agent

from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


def extract_write_to_code_cell_calls(messages):
    write_calls = []

    for msg in messages:
        # Try to access tool calls from both dicts and AIMessage-like objects
        tool_calls = []
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls", []) or msg.get("additional_kwargs", {}).get("tool_calls", [])
        else:
            tool_calls = getattr(msg, "tool_calls", []) or getattr(msg, "additional_kwargs", {}).get("tool_calls", [])

        # Loop through all tool calls in this message
        for tool_call in tool_calls:
            if tool_call.get("name") == "write_to_code_cell":
                write_calls.append(tool_call)

    return write_calls



def notebooks_are_different(current_cells, prev_cells):
    if not prev_cells:
        return True
    try:
        def simplify(cell):
            return {
                "source": cell.get("source", ""),
                "cell_type": cell.get("cell_type", ""),
                "outputs": cell.get("outputs", []),
                "execution_count": cell.get("execution_count", None),
            }

        simplified_current = [simplify(c) for c in current_cells]
        simplified_previous = [simplify(c) for c in prev_cells]

        return simplified_current != simplified_previous
    except Exception:
        return True


class State(TypedDict):
    messages: list


class LinterPersona(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_cell = ""
        self._notebooks = {}
        self._collab_task_in_progress = False
        self._global_awareness_observer = None
        self._created_agent = False
        self._results = None

    @property
    def defaults(self):
        return PersonaDefaults(
            name="LinterPersona",
            description="A Jupyter AI Assistant who can write to notebooks to add comments",
            avatar_path="/api/ai/static/jupyternaut.svg",
            system_prompt="You are a function-calling assistant operating inside a JupyterLab environment, use your tools to operate on the notebook!",
        )

    def get_active_cell(self, notebook):
        """Return the ID of the currently selected cell of a given notebook, or None if none are active."""
        awareness_states = notebook.awareness.states
        for client_id, state in awareness_states.items():
            active_cell = state.get("activeCellId")
            if active_cell:
                return active_cell
        return "NO CELL ID"

    def extract_current_notebook_path(
        self, global_awareness_doc, target_username: str
    ) -> str | None:
        """Helper to grab the path of the currently active notebook."""
        for client_id, state in global_awareness_doc.awareness.states.items():
            user = state.get("user", {})
            username = user.get("username")
            if username == target_username:
                current = state.get("current")
                if current and current.startswith("notebook:"):
                    return current.removeprefix("notebook:RTC:")
        return None

    async def get_active_notebook(
        self, client_id: str, notebook_path: str
    ) -> YNotebook | None:
        """Get the live notebook object given it's path"""
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        collaboration = serverapp.web_app.settings["jupyter_server_ydoc"]
        websocket_server = collaboration.ywebsocket_server

        for room_id in websocket_server.rooms:
            try:
                doc = await collaboration.get_document(room_id=room_id, copy=False)
                if (
                    isinstance(doc, YNotebook)
                    and getattr(doc, "path", None) == notebook_path
                ):
                    awareness_states = doc.awareness.states

                    for client_id, state in awareness_states.items():
                        active_cell = state.get("activeCellId")
                        self.log.info(
                            f"👤 Client {client_id} activeCellId: {active_cell}"
                        )

                    return doc
            except Exception as e:
                self.log.warning(f"⚠️ Could not inspect room {room_id}: {e}")

        self.log.warning(f"❌ No active notebook found for client_id: {client_id}")
        return None


    async def create_langgraph_agent(self, notebook: YNotebook):
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        extension_manager = serverapp.extension_manager
        raw_tools = find_tools(extension_manager, return_metadata_only=False)

        agent = await external_create_langgraph_agent(
            extension_manager,
            self.log,
            self.ychat,
            raw_tools,
            notebook,
            self.id,
            self.get_active_cell,
        )
        return agent
    
    async def _run_with_flag_reset(self, tone_prompt, path, notebook):
        """Run the LangGraph agent with the given prompts and clear the busy flag."""
        try:
            agent = self._notebooks[path]["agent"]
            history = self._notebooks[path]["history"]
            current_cell = self.get_active_cell(notebook)
            messages = await external_run_langgraph_agent(self.log, agent, history, tone_prompt, current_cell)
            calls = extract_write_to_code_cell_calls(messages['messages'])
            self._notebooks[path]["history"].append(calls)
        finally:
            self._collab_task_in_progress = False


    async def start_collaborative_session(self, ynotebook: YNotebook, path: str, tone_prompt):
        """
        Observes awareness (cursor position, etc) and reacts when a user changes their selection.
        """

        def on_awareness_change(event_type, data):
            # Don't cancel current job — just let it finish
            if self._collab_task_in_progress:
                return

            current_cell = self.get_active_cell(ynotebook)
            last_cell = self._notebooks[path]["activeCell"]
            
            if current_cell != last_cell:
                self._notebooks[path]["activeCell"] = current_cell
                current_notebook = [ynotebook.get_cell(i) for i in range(len(ynotebook._ycells))]
                self.log.info(f"CURRENT NOTEBOOK STATE: {current_notebook}")
                
                last_notebook = self._notebooks[path]["lastNotebookState"]
                self.log.info(f"LAST NOTEBOOK STATE: {last_notebook}")

                if notebooks_are_different(current_notebook, last_notebook):
                    self._notebooks[path]["lastNotebookState"] = current_notebook
                    self.ychat.add_message(
                        NewMessage(
                            body=f"Active Cell is now: {current_cell}",
                            sender=self.id,
                        )
                    )

                    self._collab_task_in_progress = True
                    self._running_task = asyncio.create_task(
                        self._run_with_flag_reset(tone_prompt, path, ynotebook)
                    )
        agent = await self.create_langgraph_agent(ynotebook)
        self._notebooks[path]["agent"] = agent
        self._notebooks[path]["history"] = []
        self._notebooks[path]["lastNotebookState"] = None
        awareness = ynotebook.awareness
        unsubscribe = awareness.observe(on_awareness_change)
        self._notebooks[path]["observer"] = (awareness, unsubscribe)
        self.log.info(f"✅ Awareness observer registered for notebook: {path}")

    async def _handle_global_awareness_change(self, client_id, tone_prompt):
        """Respond to a global awareness update by tracking the newly active notebook.

        If the user switches to a different notebook (based on global awareness state),
        this method detects the change, retrieves the notebook document, stores its
        active cell, and starts a new collaborative session if one is not already running.
        """
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        collaboration = serverapp.web_app.settings["jupyter_server_ydoc"]
        websocket_server = collaboration.ywebsocket_server

        the_room_id = "JupyterLab:globalAwareness"
        global_doc = websocket_server.rooms[the_room_id]

        active_notebook_path = self.extract_current_notebook_path(global_doc, client_id)
        if not active_notebook_path:
            self.log.warning("❌ No active notebook path found.")
            return

        # check if a new notebook has been clicked on
        if active_notebook_path not in self._notebooks:
            notebook = await self.get_active_notebook(client_id, active_notebook_path)
            active_cell = self.get_active_cell(notebook)
            self._notebooks[active_notebook_path] = {
                "activeCell": active_cell,
                "observer": None,
            }
            if notebook:
                await self.start_collaborative_session(
                    notebook, active_notebook_path, tone_prompt
                )
            else:
                self.log.info(
                    f"THERE WAS NO COLLABORATIVE NOTEBOOK OBSERVER STARTED FOR {active_notebook_path}"
                )
    async def start_global_observation(self, client_id, tone_prompt):
        """
        Observes awareness changes in global awarness
        """
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        collaboration = serverapp.web_app.settings["jupyter_server_ydoc"]
        websocket_server = collaboration.ywebsocket_server

        the_room_id = "JupyterLab:globalAwareness"

        doc = websocket_server.rooms[the_room_id]

        def on_awareness_change(event_type, data):
            asyncio.create_task(
                self._handle_global_awareness_change(client_id, tone_prompt)
            )

        awareness = doc.awareness
        unsubscribe = awareness.observe(on_awareness_change)
        self._global_awareness_observer = (awareness, unsubscribe)
        self.log.info("✅ GLOBAL Awareness observer registered.")

    async def process_message(self, message: Message):
        """
        Set up a basic supervising agent to start and stop a collaborative session with live notebook editing agent.
        """
        client_id = message.sender

        @tool
        async def start_collaborative_session() -> str:
            """Starts a comment adding collaborative session. Optionally accepts a tone prompt."""
            await self.start_global_observation(client_id, "")
            return f"Collaborative session started."

        @tool
        async def stop_collaborative_session() -> str:
            """Stops the current collaborative comment adding session."""

            # Unregister global awareness observer
            if self._global_awareness_observer:
                awareness, unsubscribe = self._global_awareness_observer
                awareness.unobserve(unsubscribe)
                self._global_awareness_observer = None
                self.log.info("🛑 Global awareness observer removed.")

            # Unregister all notebook-level observers
            for path, info in self._notebooks.items():
                awareness, unsubscribe = info["observer"]
                awareness.unobserve(unsubscribe)
                self.log.info(f"🛑 Notebook awareness observer removed for: {path}")

            # reset per-notebook state
            self._notebooks.clear()

            return "Collaborative session stopped and all observers removed."

        async def stream_typing(full_text: str) -> str:
            """Streams a chat message to the user as if it's being typed."""
            stream_msg_id = self.ychat.add_message(NewMessage(body="", sender=self.id))
            current_text = ""

            for char in full_text:
                await asyncio.sleep(0.02)
                current_text += char
                self.ychat.update_message(
                    Message(
                        id=stream_msg_id,
                        body=current_text,
                        time=time(),
                        sender=self.id,
                        raw_time=False,
                    ),
                    append=False,
                )
            return "Typing stream completed."

        tools = [start_collaborative_session, stop_collaborative_session]
        if self._created_agent == False: 
            self.supervisor_agent = await create_supervisor_agent(
                logger=self.log,
                stream=stream_typing,
                tools=tools,
            )
            self._created_agent = True
            self.log.info("CREATING NEW AGENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!---------------------------")
        
        result = await run_supervisor_agent(self.log, self.supervisor_agent, message.body, self._results)
        self.log.info(f"RESULT: {result}")
        self._results = result

