import json
import os
import sqlite3
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
    """Simple in-memory storage for question and answer traces."""

    def __init__(self):
        self.question_trace = []
        self.answer_trace = []

    def add_interaction(self, question, answer):
        self.question_trace.append(question)
        self.answer_trace.append(answer)

    def get_context(self):
        if not self.question_trace:
            return ""
        else:
            context_lines = [
                "Here are the questions and answers from the previous interactions.",
                "Use them to answer the current question if they are relevant:",
            ]
            for q, a in zip(self.question_trace, self.answer_trace):
                context_lines.append(f"QUESTION: {q}")
                context_lines.append(f"ANSWER: {a}")
            return "\n".join(context_lines)


# Pydantic Models for output validation
class DecomposedQuestion(BaseModel):
    sub_questions: List[str]


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
        db_path="./sql_lite_database.db",
        memory_path="agent_memory.json",
    ):
        """Initialize Agent with database path and model."""
        self.model = model
        self.client = UnifiedChatAPI(model=self.model)
        self.memory = self.load_memory()
        self.context = ""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect_db()
        self.memory_path = memory_path

    # Database Management
    def _connect_db(self):
        """Connect to the SQLite database."""
        if not os.path.exists(self.db_path):
            raise RuntimeError(f"Database file not found at: {self.db_path}")
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            self._close_db()
            raise RuntimeError(f"Database connection failed: {e}")

    def _close_db(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.cursor = None
        self.conn = None

    def __del__(self):
        """Destructor to ensure the database connection is closed."""
        self._close_db()

    # Memory Management
    def load_memory(self):
        """Load the agent memory from a JSON file."""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return SimpleMemory()

    def save_memory(self):
        """Save the agent memory to a JSON file."""
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "question_trace": self.memory.question_trace,
                    "answer_trace": self.memory.answer_trace,
                },
                f,
                indent=4,
            )

    # Agent Reflections
    def reflection(self, question: str) -> str:
        """Perform an agent reflection."""
        context = self.context or "<No previous questions have been asked>"
        agent_template = f"""CONTEXTUAL INFORMATION
{context}

QUESTION
{question}"""

        assistant_reply = self.client.chat(
            [
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": agent_template},
            ]
        )
        return assistant_reply

    # Agent Actions
    def action(
        self, question: str, recursion=False, max_retrials: int = 3
    ) -> AgentAction:
        """Determine the next action for the agent."""
        action_system_prompt = (
            ACTION_SYSTEM_PROMPT_01
            + (not recursion) * ACTION_SYSTEM_PROMPT_DECOMPOSITION
            + ACTION_SYSTEM_PROMPT_02
        )

        context = self.context or "<No previous questions have been asked>"
        agent_template = f"""CONTEXTUAL INFORMATION
{context}

QUESTION
{question}"""

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
        self, question: str, recursion: bool = False, indent_level: int = 0
    ) -> str:
        """Run the ReAct agent to answer a question."""
        if not recursion:
            self.context = self.memory.get_context()
            print("\n")

        while True:
            try:
                self.perform_reflection(question, indent_level)
                action = self.decide_action(question, recursion, indent_level)
                result = self.execute_action(
                    action, question, recursion, indent_level
                )

                if result is not None:
                    return result

            except Exception as e:
                error_msg = self.format_message(str(e), "ERROR", indent_level)
                self.context += error_msg
                break

    # Helper Methods
    def perform_reflection(self, question: str, indent_level: int):
        """Perform reflection and update context."""
        reflection = self.reflection(question=question)
        reflection_msg = self.format_message(
            reflection.split(">> ")[1], "REFLECTION", indent_level
        )
        self.context += reflection_msg

    def decide_action(
        self,
        question: str,
        recursion: bool,
        indent_level: int,
        max_retrials: int = 3,
    ) -> AgentAction:
        """Decide on the next action and update context."""
        action = self.action(
            question=question, recursion=recursion, max_retrials=max_retrials
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
        self, action: AgentAction, question: str, indent_level: int
    ) -> Optional[str]:
        """Execute the chosen action and handle the result."""
        try:
            result = None
            if action.request == "observe_book":
                result = self.observe_book(int(action.argument))
            elif action.request == "run_main":
                result = self.run_main()
            elif action.request == "observe_repository":
                result = self.observe_repository()
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
                self.handle_final_answer(question, action, indent_level)
                return action.argument
            else:
                raise ValueError(f"Unknown action request: {action.request}")

            if result is not None:
                obs_msg = self.format_message(
                    str(result), "OBSERVATION", indent_level
                )
                self.context += obs_msg
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
        result = self.decompose_question(question=action.argument)
        obs_msg = self.format_message(str(result), "OBSERVATION", indent_level)
        self.context += obs_msg

        # Answer subquestions recursively
        answers = []
        for subquestion in result.sub_questions:
            subq_msg = self.format_message(
                subquestion, "SUBQUESTION", indent_level
            )
            self.context += subq_msg
            # Run agent recursively
            answer = self.run_agent(
                subquestion,
                recursion=True,
                indent_level=min(indent_level + 1, 3),
            )
            answers.append(answer)

        # Summarize answers
        summary = self.answers_summarizer(result.sub_questions, answers)
        summary_msg = self.format_message(
            summary.summary, "GENERATED RESPONSE TO SUBQUESTIONS", indent_level
        )
        self.context += summary_msg

    # Assistants
    def decompose_question(
        self, question: str, max_retrials: int = 3
    ) -> DecomposedQuestion:
        """Decompose a complex question into simpler parts."""
        decomp_system_prompt = """GENERAL INSTRUCTIONS
You are an expert in the domain of the following question. Your task is to decompose a complex question into simpler parts.

RESPONSE FORMAT
{"sub_questions":["<FILL>"]}"""

        for attempt in range(max_retrials):
            assistant_reply = self.client.chat(
                [
                    {"role": "system", "content": decomp_system_prompt},
                    {"role": "user", "content": question},
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

    def answers_summarizer(
        self, questions: List[str], answers: List[str], max_retrials: int = 3
    ) -> AnswersSummary:
        """Summarize a list of answers to the decomposed questions."""
        answer_summarizer_system_prompt = """GENERAL INSTRUCTIONS
You are an expert in the domain of the following questions. Your task is to summarize the answers to the questions into a single response.

RESPONSE FORMAT
{"summary": "<FILL>"}"""

        q_and_a_prompt = "\n\n".join(
            [
                f"SUBQUESTION {i+1}\n{q}\nANSWER {i+1}\n{a}"
                for i, (q, a) in enumerate(zip(questions, answers))
            ]
        )

        for attempt in range(max_retrials):
            assistant_reply = self.client.chat(
                [
                    {
                        "role": "system",
                        "content": answer_summarizer_system_prompt,
                    },
                    {"role": "user", "content": q_and_a_prompt},
                ]
            )

            try:
                response_content = json.loads(assistant_reply)
                validated_response = AnswersSummary.model_validate(
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

    # Final Answer Tool
    def handle_final_answer(
        self, question: str, action: AgentAction, indent_level: int
    ):
        """Handle the final answer action."""
        # Update memory
        self.memory.add_interaction(question, action.argument)
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
            "REFLECTION": "\033[94m",  # Blue
            "ACTION": "\033[92m",  # Green
            "OBSERVATION": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
            "SUBQUESTION": "\033[95m",  # Magenta
            "FINAL ANSWER": "\033[96m",  # Cyan
            "ARGUMENT": "\033[90m",  # Gray
            "GENERATED RESPONSE TO SUBQUESTIONS": "\033[96m",  # Cyan
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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    GPT_MODEL = "gpt-4o-mini"
    OLLAMA_MODEL = "qwen2.5-coder:7b"

    SELECTED_MODEL = OLLAMA_MODEL

    if SELECTED_MODEL == GPT_MODEL:
        agent = AgentReAct(
            model=SELECTED_MODEL,
            db_path="sql_lite_database.db",
            memory_path="agent_memory_gpt.json",
        )
        question = "How did sales vary between Q1 and Q2 of 2024 in percentage and amount?"
        agent.run_agent(question)
        agent.save_context_to_html("agent_context_gpt.html")
        agent.save_memory()

    elif SELECTED_MODEL == OLLAMA_MODEL:
        agent = AgentReAct(
            model=SELECTED_MODEL,
            db_path="sql_lite_database.db",
            memory_path="agent_memory_ollama.json",
        )
        simpler_question = "How many orders were there in 2024?"
        agent.run_agent(simpler_question)
        agent.save_context_to_html("agent_context_ollama.html")
