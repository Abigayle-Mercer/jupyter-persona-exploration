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
from langchain_core.tools import tool

from .agent import run_supervisor_agent

from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


class State(TypedDict):
    messages: list


class GrammarEditor(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_cell = ""
        self._notebooks = {}
        self._collab_task_in_progress = False
        self._global_awareness_observer = None
        self._edited_cells = []

    @property
    def defaults(self):
        return PersonaDefaults(
            name="GrammarEditor",
            description="A Jupyter AI Assistant who can write to notebooks to fix grammar mistakes",
            avatar_path="/api/ai/static/jupyternaut.svg",
            system_prompt="You are a function-calling assistant operating inside a JupyterLab environment, use your tools to operate on the notebook!",
        )

    def set_edited_cells(self, cell):
        """Record that the agent has edited the given cell."""
        self._edited_cells.append(cell)

    def get_edited_cells(self):
        """Return the list of cell IDs that have already been edited."""
        return self._edited_cells

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

    async def run_langgraph_agent(
        self, notebook: YNotebook, user_prompt: str, tone_prompt
    ):
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        handler = CallContext.get(CallContext.JUPYTER_HANDLER)
        serverapp = handler.serverapp
        extension_manager = serverapp.extension_manager
        raw_tools = find_tools(extension_manager, return_metadata_only=False)

        await external_run_langgraph_agent(
            extension_manager,
            self.log,
            self.ychat,
            raw_tools,
            notebook,
            user_prompt,
            tone_prompt,
            self.id,
            self.get_active_cell,
            self.set_edited_cells,
            self.get_edited_cells,
        )

    async def _run_with_flag_reset(self, ynotebook, prompt, tone_prompt):
        """Run the LangGraph agent with the given prompts and clear the busy flag."""
        try:
            await self.run_langgraph_agent(ynotebook, prompt, tone_prompt)
        finally:
            self._collab_task_in_progress = False

    def start_collaborative_session(self, ynotebook: YNotebook, path: str, tone_prompt):
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
                self.ychat.add_message(
                    NewMessage(
                        body=f"Active Cell is now: {current_cell}",
                        sender=self.id,
                    )
                )

                prompt = f"""The user is currently editing in cell {current_cell} so, to avoid disrupting their work, DO NOT write to, or delete that cell. 
                Additionally, the following cells have already been edited by you, the agent, and thus SHOULD NOT be written to: {self._edited_cells}
                DO NOT UNDER ANY CIRCUMSTANCES WRITE TO THE CELLS THAT HAVE BEEN ALREADY EDITED BY YOU. ANY CELL IN THAT LIST IS A CELL THAT WAS PREVIOSLY EDITED BY YOU. 
                You can only edit in the rest of the notebook cells"""

                self._collab_task_in_progress = True
                self._running_task = asyncio.create_task(
                    self._run_with_flag_reset(ynotebook, prompt, tone_prompt)
                )

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
                self.start_collaborative_session(
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
        async def start_collaborative_session(tone_prompt) -> str:
            """Starts a comment adding collaborative session. Optionally accepts a tone prompt for the use of the agent that will edit the notebook.
            Ex. friendly and casual."""
            await self.start_global_observation(client_id, tone_prompt)
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
        await run_supervisor_agent(
            logger=self.log,
            stream=stream_typing,
            user_message=message.body,
            tools=tools,
        )
