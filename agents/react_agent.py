import json
import os
import shutil
import subprocess
import sys
import textwrap
from typing import List, Optional

import ollama
import openai
from ansi2html import Ansi2HTMLConverter
from prompts import (
    ACTION_SYSTEM_PROMPT_01,
    ACTION_SYSTEM_PROMPT_02,
    ACTION_SYSTEM_PROMPT_DECOMPOSITION,
    REFLECTION_SYSTEM_PROMPT,
)
from pydantic import BaseModel, ValidationError


# Unified Chat API
class UnifiedChatAPI:
    """Unified interface for OpenAI and Ollama chat APIs."""

    def __init__(self, model="gpt-4o-mini", openai_api_key=None):
        self.model = model
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.api = self._determine_api()
        if self.api == "openai":
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key must be provided for OpenAI models."
                )
            else:
                self.client = openai.OpenAI(api_key=self.api_key)
        elif self.api == "ollama":
            self.client = None

    def _determine_api(self):
        """Determine the API based on the model name."""
        if self.model.startswith("gpt-") or self.model.startswith("o1-"):
            return "openai"
        else:
            return "ollama"

    def chat(self, messages):
        """Wrapper for chat API."""
        if self.api == "openai":
            return self._openai_chat(messages)
        elif self.api == "ollama":
            return self._ollama_chat(messages)
        else:
            raise ValueError(
                "Unsupported API. Please set the API to 'openai' or 'ollama'."
            )

    def _openai_chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content

    def _ollama_chat(self, messages):
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]


class SimpleMemory:
    """Simple in-memory storage for task and answer traces."""

    def __init__(self, task_trace: List[str] = [], answer_trace: List[str] = []):
        self.task_trace = task_trace
        self.answer_trace = answer_trace

    def add_interaction(self, task, answer):
        self.task_trace.append(task)
        self.answer_trace.append(answer)

    def get_context(self):
        if not self.task_trace:
            return ""
        else:
            context_lines = [
                "Here are the tasks and answers from the previous interactions.",
                "Use them to answer the current task if they are relevant:",
            ]
            for q, a in zip(self.task_trace, self.answer_trace):
                context_lines.append(f"TASK: {q}")
                context_lines.append(f"ANSWER: {a}")
            return "\n".join(context_lines)


# Pydantic Models for output validation
class DecomposedQuestion(BaseModel):
    sub_tasks: List[str]


class AgentAction(BaseModel):
    request: str
    argument: Optional[str]


class AnswersSummary(BaseModel):
    summary: str


# Big Agent Class
class AgentReAct:
    """Agent class implementing the ReAct framework."""

    def __init__(
        self,
        model="gpt-4o-mini",
        memory_path="agent_memory.json",
    ):
        """Initialize Agent with database path and model."""
        self.model = model
        self.client = UnifiedChatAPI(model=self.model)
        self.context = ""
        self.memory_path = memory_path
        self.large_observations = []
        self.memory = self.load_memory()

    # Memory Management
    def load_memory(self):
        """Load the agent memory from a JSON file."""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                memory = json.load(f)
                return SimpleMemory(
                    task_trace=memory["task_trace"],
                    answer_trace=memory["answer_trace"],
                )
        else:
            return SimpleMemory()

    def save_memory(self):
        """Save the agent memory to a JSON file."""
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_trace": self.memory.task_trace,
                    "answer_trace": self.memory.answer_trace,
                },
                f,
                indent=4,
            )

    # Agent Reflections
    def reflection(self, task: str) -> str:
        """Perform an agent reflection."""
        context = self.context or "<No previous tasks have been asked>"
        agent_template = f"""CONTEXTUAL INFORMATION
{context}

TASK
{task}"""

        assistant_reply = self.client.chat(
            [
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": agent_template},
            ]
        )
        return assistant_reply

    # Agent Actions
    def action(
        self, task: str, recursion=False, max_retrials: int = 3
    ) -> AgentAction:
        """Determine the next action for the agent."""
        action_system_prompt = (
            ACTION_SYSTEM_PROMPT_01
            + (not recursion) * ACTION_SYSTEM_PROMPT_DECOMPOSITION
            + ACTION_SYSTEM_PROMPT_02
        )

        context = self.context or "<No previous tasks have been asked>"
        agent_template = f"""CONTEXTUAL INFORMATION
{context}

TASK
{task}"""

        for attempt in range(max_retrials):
            assistant_reply = self.client.chat(
                [
                    {"role": "system", "content": action_system_prompt},
                    {"role": "user", "content": agent_template},
                ]
            )

            try:
                # Attempt to extract the JSON object from the assistant's reply
                start_index = assistant_reply.find("{")
                end_index = assistant_reply.rfind("}") + 1
                json_str = assistant_reply[start_index:end_index]
                agent_action = json.loads(json_str)
                validated_response = AgentAction.model_validate(agent_action)
                return validated_response
            except (json.JSONDecodeError, ValidationError) as e:
                error_msg = self.format_message(
                    f"Validation error on attempt {attempt + 1}: {e}",
                    "ERROR",
                    0,
                )
                print(
                    f"Assistant reply on attempt {attempt + 1}:\n{assistant_reply}\n"
                )
                self.context += error_msg
                # Provide feedback to the assistant about the error
                agent_template += (
                    "\n\nERROR >> The previous response was not valid JSON or did not follow the expected format."
                    " Please respond with a valid JSON object matching the required format."
                )
                continue

        raise RuntimeError(
            "Maximum number of retries reached without successful validation."
        )

    def run_agent(
        self, task: str, recursion: bool = False, indent_level: int = 0
    ) -> str:
        """Run the ReAct agent to solve a task."""
        if not recursion:
            self.context = self.memory.get_context()
            print("\n")

        while True:
            try:
                self.perform_reflection(task, indent_level)
                action = self.decide_action(task, recursion, indent_level)
                result = self.execute_action(
                    action, task, indent_level
                )

                if result is not None:
                    self.summarize_large_observations()
                    return result

            except Exception as e:
                error_msg = self.format_message(str(e), "ERROR", indent_level)
                self.context += error_msg
                break

    # Helper Methods
    def perform_reflection(self, task: str, indent_level: int):
        """Perform reflection and update context."""
        reflection = self.reflection(task=task)
        reflection_msg = self.format_message(
            reflection.split(">> ")[1], "REFLECTION", indent_level
        )
        self.context += reflection_msg

    def decide_action(
        self,
        task: str,
        recursion: bool,
        indent_level: int,
        max_retrials: int = 3,
    ) -> AgentAction:
        """Decide on the next action and update context."""
        action = self.action(
            task=task, recursion=recursion, max_retrials=max_retrials
        )
        action_msg = self.format_message(
            action.request, "ACTION", indent_level
        )
        self.context += action_msg
        if action.argument:
            arg_msg = self.format_message(
                action.argument, "ARGUMENT", indent_level
            )
            self.context += arg_msg
        os.system("cls" if os.name == "nt" else "clear")
        print(self.context)
        return action

    def execute_action(
        self, action: AgentAction, task: str, indent_level: int
    ) -> Optional[str]:
        """Execute the chosen action and handle the result."""
        try:
            result = None
            large_output = False

            if action.request == "observe_book":
                result = self.observe_book(int(action.argument))
            elif action.request == "run_main":
                result = self.run_main()
            elif action.request == "observe_repository":
                result = self.observe_repository()
                large_output = True
            elif action.request == "observe_single_script":
                result = self.observe_single_script(action.argument)
                large_output = True
            elif action.request == "insert_code":
                args = json.loads(action.argument)
                result = self.insert_code(
                    args["file_path"], args["row"], args["code"]
                )
            elif action.request == "modify_code":
                args = json.loads(action.argument)
                result = self.modify_code(
                    args["file_path"],
                    args["begin_row"],
                    args["end_row"],
                    args["code"],
                )
            elif action.request == "rewrite_script":
                args = json.loads(action.argument)
                result = self.rewrite_script(args["file_path"], args["code"])
            elif action.request == "decomposition":
                self.handle_decomposition(action, indent_level)
                return None
            elif action.request == "final_answer":
                self.handle_final_answer(task, action, indent_level)
                return action.argument
            else:
                raise ValueError(f"Unknown action request: {action.request}")

            if result is not None:
                obs_msg = self.format_message(
                    str(result), "OBSERVATION", indent_level
                )
                self.context += obs_msg
                if large_output:
                    self.large_observations.append((obs_msg, indent_level))

        except Exception as e:
            error_msg = self.format_message(
                f"Error executing {action.request}: {str(e)}",
                "ERROR",
                indent_level,
            )
            self.context += error_msg
        return None

    def handle_decomposition(self, action: AgentAction, indent_level: int):
        """Handle the decomposition action."""
        result = self.decompose_task(task=action.argument)
        obs_msg = self.format_message(str(result), "OBSERVATION", indent_level)
        self.context += obs_msg

        # Answer subtasks recursively
        answers = []
        for subtask in result.sub_tasks:
            subq_msg = self.format_message(subtask, "SUBTASK", indent_level)
            self.context += subq_msg
            # Run agent recursively
            answer = self.run_agent(
                subtask,
                recursion=True,
                indent_level=min(indent_level + 1, 3),
            )
            answers.append(answer)

    # Assistants
    def decompose_task(
        self, task: str, max_retrials: int = 3
    ) -> DecomposedQuestion:
        """Decompose a complex task into simpler parts."""
        decomp_system_prompt = """GENERAL INSTRUCTIONS
You are an expert in the domain of the following task. Your task is to decompose a complex task into simpler parts.

RESPONSE FORMAT
{"sub_tasks":["<FILL>"]}"""

        for attempt in range(max_retrials):
            assistant_reply = self.client.chat(
                [
                    {"role": "system", "content": decomp_system_prompt},
                    {"role": "user", "content": task},
                ]
            )

            try:
                response_content = json.loads(assistant_reply)
                validated_response = DecomposedQuestion.model_validate(
                    response_content
                )
                return validated_response
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Validation error on attempt {attempt + 1}: {e}")

        raise RuntimeError(
            "Maximum number of retries reached without successful validation."
        )

    # Tools
    def observe_book(self, section: int) -> Optional[str]:
        """Extracts content from a specified section of the book."""
        try:
            toc = {
                1: "Overview",
                2: "Motion Blur",
                3: "Bounding Volume Hierarchies",
                4: "Texture Mapping",
                5: "Perlin Noise",
                6: "Quadrilaterals",
                7: "Lights",
                8: "Instances",
                9: "Volumes",
                10: "A Scene Testing All New Features",
            }

            with open("RayTracingTheNextWeek.html", encoding="utf-8") as f:
                book = f.read()
                search_term = f"{toc[section]}\n=="
                start_index = book.find(search_term)
                if start_index == -1:
                    return "Section not found."

                end_index = book.find(f"{toc[section+1]}\n==", start_index)
                if end_index == -1:
                    end_index = len(book)

                return book[start_index:end_index]
        except Exception as e:
            print(f"Error observing book section: {e}")
            return None

    def run_main(self) -> Optional[str]:
        """Execute the main script."""
        try:
            result = subprocess.run(
                [sys.executable, "src/main.py"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running main: {e.stderr}")
            return None

    def observe_repository(self) -> Optional[str]:
        """Get concatenated content of all Python files in src/."""
        try:
            src_path = "src/"
            concatenated_content = ""

            for root, _, files in os.walk(src_path):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            concatenated_content += (
                                f"### {file_path}\n# "
                                + "=" * 98
                                + "\n"
                                + f.read()
                                + "\n\n"
                            )

            return concatenated_content
        except Exception as e:
            print(f"Error observing repository: {e}")
            return None

    def insert_code(
        self, file_path: str, row: int, code: str
    ) -> Optional[str]:
        """Insert code at specific row."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            lines.insert(row, code)

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return "Code inserted successfully"
        except Exception as e:
            print(f"Error inserting code: {e}")
            return None

    def modify_code(
        self, file_path: str, begin_row: int, end_row: int, code: str
    ) -> Optional[str]:
        """Modify code between specific rows."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            lines[begin_row:end_row] = [code]

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return "Code modified successfully"
        except Exception as e:
            print(f"Error modifying code: {e}")
            return None

    def rewrite_script(self, file_path: str, code: str) -> Optional[str]:
        """Rewrite entire file content."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            return "Script rewritten successfully"
        except Exception as e:
            print(f"Error rewriting script: {e}")
            return None

    def observe_single_script(self, script_name: str) -> Optional[str]:
        """Get content of a specific script."""
        try:
            with open(f"src/{script_name}", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error observing script: {e}")
            return None

    # Final Answer Tool
    def handle_final_answer(
        self, task: str, action: AgentAction, indent_level: int
    ):
        """Handle the final answer action."""
        # Update memory
        self.memory.add_interaction(task, action.argument)
        final_answer_msg = self.format_message(
            action.argument, "FINAL ANSWER", indent_level
        )
        self.context += final_answer_msg
        os.system("cls" if os.name == "nt" else "clear")
        print(self.context)

    # Formatting
    def format_message(self, text: str, action: str, indent_level: int) -> str:
        """Format messages with indentation and color."""
        indent = "    " * indent_level
        colored_action = self.color_text(f"{action} >> ", action)
        wrapped_text = textwrap.fill(text, width=100)
        indented_text = textwrap.indent(
            wrapped_text, "    " * (indent_level + 1)
        )
        return f"{indent}{colored_action}{indented_text}\n"

    def color_text(self, text: str, action: str) -> str:
        """Colorize text based on the action."""
        color_codes = {
            "REFLECTION": "\033[94m",
            "ACTION": "\033[92m",
            "OBSERVATION": "\033[93m",
            "ERROR": "\033[91m",
            "SUBTASK": "\033[95m",
            "FINAL ANSWER": "\033[96m",
            "ARGUMENT": "\033[90m",
            "GENERATED RESPONSE TO SUBTASKS": "\033[96m",
            "SUMMARY": "\033[97m",
        }
        reset_code = "\033[0m"
        color_code = color_codes.get(action, "")
        return f"{color_code}{text}{reset_code}"

    # Saving Trace of Thought
    def save_context_to_html(self, filename="agent_context.html"):
        """Save the agent context to an HTML file."""
        conv = Ansi2HTMLConverter()
        html_content = conv.convert(self.context, full=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Context saved to {filename}")

    def summarize_large_observations(self):
        """Summarize large observations for context management."""
        for obs_msg, indent_level in self.large_observations:
            summary = self.summarize_text(obs_msg)
            summary_msg = self.format_message(summary, "SUMMARY", indent_level)
            self.context = self.context.replace(obs_msg, summary_msg)
        self.large_observations.clear()

    def summarize_text(self, text: str) -> str:
        """Summarize technical content."""
        system_prompt = (
            "You are an assistant that summarizes technical content."
        )
        response = self.client.chat(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Please provide a concise summary of the following code using natural language:\n{text}",
                },
            ]
        )
        return response.strip()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    if os.path.exists("src/"):
        shutil.rmtree("src/")
    shutil.copytree("src_original/", "src/")

    GPT_MODEL = "gpt-4o-mini"
    OLLAMA_MODEL = "qwen2.5-coder:7b"

    SELECTED_MODEL = GPT_MODEL

    task = """
I have an implementation of the ray tracer algorithm in PyTorch.
It is based on the book 'Ray Tracing In One Weekend', where the code is written in C++.

I want you to augment my code with additional features as described in the book 'Ray Tracing The Next Week', 
the second book of the series. You will be evaluated on the completeness and correctness of the implementation.
At the end of the task, I will personally review the code and evaluate your work. The nine scenes that are rendered
in the book must be rendered correctly.

I recommend you to start the task by using the observe_repository tool to get a view of the current code.
Use the same style and structure as in the currently implemented code.
"""

    if SELECTED_MODEL == GPT_MODEL:
        agent = AgentReAct(
            model=SELECTED_MODEL,
            memory_path="agent_memory_gpt.json",
        )
        agent.run_agent(task)
        agent.save_context_to_html("agent_context_gpt.html")
        agent.save_memory()

    elif SELECTED_MODEL == OLLAMA_MODEL:
        agent = AgentReAct(
            model=SELECTED_MODEL,
            memory_path="agent_memory_ollama.json",
        )
        agent.run_agent(task)
        agent.save_context_to_html("agent_context_ollama.html")
        agent.save_memory()
