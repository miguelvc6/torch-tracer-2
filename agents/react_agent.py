import json
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from typing import List, Optional

import ollama
import openai
from ansi2html import Ansi2HTMLConverter

from pydantic import BaseModel, ValidationError

# System prompts for the language model
REFLECTION_SYSTEM_PROMPT = """GENERAL INSTRUCTIONS
You must reflect on the proposed task and use the context to decide how to solve it.
You must decide whether to use a tool or end the task if all the requirements have already been fulfilled.
Write a brief reflection with the indicated response format.
Do not call any actions or tools, return only the reflection.

AVAILABLE TOOLS
- observe_book: {"Description": "Extracts content from a specified section of the book RayTracingTheNextWeek. \
    The input is the section number. The following is the corresponding table of contents:
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
    ", "Arguments": section - int}
- observe_single_script: {"Description": "Get content of a specific python script from the repository", "Arguments": script_name - str}
- run_main: {
    "Description": "Execute the main python script in src/main.py. The console output of the script is returned, \
    may it be print statements, other outputs or error traces.", "Arguments": None}
- observe_repository: {
    "Description": "Get concatenated content of all Python files in src/ that build up the project", 
    "Arguments": None}
- insert_code: {"Description": "Insert code at specific row in a python script", "Arguments": {"file_path": str, "row": int, "code": str}}
- modify_code: {
    "Description": "Modify code between specific rows in a python script", 
    "Arguments": {"file_path": str, "begin_row": int, "end_row": int, "code": str}
    }
- rewrite_script: {
    "Description": "Rewrite entire python file content. Attention: This will overwrite the file removing all previous content.", 
    "Arguments": {"file_path": str, "code": str}
    }

AVAILABLE ACTION
- end_task: {
    "Description": "The task is complete. You have solved all the user's requirements and the repository is complete.", 
    "Arguments": null
    }

RESPONSE FORMAT
REFLECTION >> <Fill>
"""

ACTION_SYSTEM_PROMPT = """GENERAL INSTRUCTIONS
You must solve the proposed software engineering task. Use the context and the prevoious reflection to decide how to proceed in the next step to solve the task.
You must decide whether to use a tool or give the final answer, and return a response following the response format.
If you already have enough information, you should provide a final answer by useing the end_task action. Do this only when you have fulfilled all the requirements.
Fill with null where no tool or action is required.

IMPORTANT:
- The response must be in valid JSON format.
- Ensure all text strings are properly escaped.

AVAILABLE TOOLS
- observe_book: {"Description": "Extracts content from a specified section of the book RayTracingTheNextWeek. \
    The input is the section number. The following is the corresponding table of contents:
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
    ", "Arguments": section - int}
- observe_single_script: {"Description": "Get content of a specific script from the repository", "Arguments": script_name - str}
- run_main: {
    "Description": "Execute the main script in src/main.py. The console output of the script is returned, \
    may it be print statements, other outputs or error traces.", "Arguments": None}
- observe_repository: {
    "Description": "Get concatenated content of all Python files in src/ that build up the project", 
    "Arguments": None}
- insert_code: {"Description": "Insert code at specific row in a python script", "Arguments": {"file_path": str, "row": int, "code": str}}
- modify_code: {
    "Description": "Modify code between specific rows in a python script", 
    "Arguments": {"file_path": str, "begin_row": int, "end_row": int, "code": str}
    }
- rewrite_script: {
    "Description": "Rewrite entire file content. Attention: This will overwrite the file removing all previous content.", 
    "Arguments": {"file_path": str, "code": str}
    }

AVAILABLE ACTION
- end_task: {
    "Description": "The task is complete. You have solved all the user's requirements and the repository is complete.", 
    "Arguments": null
    }

RESPONSE FORMAT
{
  "request": "<Fill>",
  "argument": "<Fill or null>"
}

EXAMPLES:

1. Using observe_book with a section number:
{
  "request": "observe_book",
  "argument": "1"
}

2. Using run_main without arguments:
{
  "request": "run_main",
  "argument": null
}

3. Using modify_code with file path and row numbers:
{
  "request": "modify_code",
  "argument": "{\"file_path\": \"src/main.py\", \"begin_row\": 10, \"end_row\": 15, \"code\": \"def new_function():\\n    return True\\n\"}"
}

4. Using observe_repository without arguments:
{
  "request": "observe_repository",
  "argument": null
}

5. Using insert_code to add new code:
{
  "request": "insert_code",
  "argument": "{\"file_path\": \"src/camera.py\", \"row\": 25, \"code\": \"    def get_focal_length(self):\\n        return self._focal_length\\n\"}"
}

6. Using rewrite_script to replace entire file:
{
  "request": "rewrite_script",
  "argument": "{\"file_path\": \"src/utils.py\", \"code\": \"def calculate_vector(x, y, z):\\n    return Vector3(x, y, z)\\n\"}"
}

7. Using decomposition for complex questions:
{
  "request": "decomposition",
  "argument": "How does the ray tracer handle both motion blur and texture mapping?"
}

8. Using observe_single_script to read a specific script:
{
  "request": "observe_single_script",
  "argument": "camera.py"
}

9. Task completed and objective fulfilled:
{
  "request": "end_task",
  "argument": null
}
"""


class UnifiedChatAPI:
    """Unified interface for OpenAI and Ollama chat APIs."""

    def __init__(
        self, model="gpt-4o-mini", openai_api_key=None, verbose=False
    ):
        self.model = model
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.api = self._determine_api()
        self.verbose = verbose
        self.date_hour = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize logging
        if self.verbose:
            os.makedirs("logs/", exist_ok=True)
            self.log_file = f"logs/chat_log_{self.date_hour}.json"
            self.chat_history = {
                "metadata": {
                    "model": self.model,
                    "api": self.api,
                    "start_time": self.date_hour,
                },
                "interactions": [],
            }
        else:
            self.log_file = None
            self.chat_history = None

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

    def log_interaction(self, messages, answer):
        """Log the interaction to the log file with proper formatting."""
        if not self.verbose:
            return

        # Format the interaction
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "messages": self._format_messages(messages),
            "response": self._format_response(answer),
        }

        # Add to chat history
        self.chat_history["interactions"].append(interaction)

        # Write to file with proper formatting
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def _format_messages(self, messages):
        """Format messages for logging."""
        formatted_messages = []
        for msg in messages:
            formatted_msg = {
                "role": msg["role"],
                "content": self._format_content(msg["content"]),
            }
            formatted_messages.append(formatted_msg)
        return formatted_messages

    def _format_response(self, response):
        """Format the response for logging."""
        return self._format_content(response)

    def _format_content(self, content):
        """Format content string with proper line breaks and escaping."""
        if not isinstance(content, str):
            content = str(content)

        # Replace literal newlines with actual newlines while preserving formatting
        content = content.replace("\\n", "\n")

        # Remove any redundant escape characters
        content = content.replace("\\\\", "\\")

        return content

    def chat(self, messages):
        """Wrapper for chat API with logging."""
        answer = None
        try:
            if self.api == "openai":
                answer = self._openai_chat(messages)
            elif self.api == "ollama":
                answer = self._ollama_chat(messages)
            else:
                raise ValueError(
                    "Unsupported API. Please set the API to 'openai' or 'ollama'."
                )

            if self.verbose:
                self.log_interaction(messages, answer)

            return answer

        except Exception as e:
            if self.verbose:
                self.log_interaction(messages, f"Error: {str(e)}")
            raise e

    def _openai_chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content

    def _ollama_chat(self, messages):
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]


# Pydantic Models for output validation
class DecomposedQuestion(BaseModel):
    sub_tasks: List[str]


class AgentAction(BaseModel):
    request: str
    argument: Optional[str]


# Big Agent Class
class AgentReAct:
    """Agent class implementing the ReAct framework."""

    def __init__(
        self,
        model="gpt-4o-mini",
        verbose=False,
    ):
        """Initialize Agent with model."""
        self.model = model
        self.client = UnifiedChatAPI(model=self.model, verbose=verbose)
        self.context = ""
        self.large_observations = []

    # Agent Reflections
    def reflection(self, task: str) -> str:
        """Perform an agent reflection."""
        context = (
            self.context or "<No previous iterations have been performed>"
        )
        agent_template = f"""PREVIOUS ITERATIONS
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
    def action(self, task: str, max_retrials: int = 3) -> AgentAction:
        """Determine the next action for the agent."""

        context = (
            self.context or "<No previous iterations have been performed>"
        )
        agent_template = f"""PREVIOUS ITERATIONS
{context}

TASK
{task}"""

        for attempt in range(max_retrials):
            assistant_reply = self.client.chat(
                [
                    {"role": "system", "content": ACTION_SYSTEM_PROMPT},
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

    def run_agent(self, task: str) -> str:
        while True:
            try:
                self.perform_reflection(task)
                action = self.decide_action(task)
                result = self.execute_action(action, task)

                if result == "end_task":
                    break
                elif result is not None:
                    # self.summarize_large_observations()
                    continue

            except Exception as e:
                error_msg = self.format_message(str(e), "ERROR")
                self.context += error_msg
                break

    # Helper Methods
    def perform_reflection(self, task: str):
        """Perform reflection and update context."""
        reflection = self.reflection(task=task)
        reflection_msg = self.format_message(
            reflection.split(">> ")[1], "REFLECTION"
        )
        self.context += reflection_msg

    def decide_action(
        self,
        task: str,
        max_retrials: int = 3,
    ) -> AgentAction:
        """Decide on the next action and update context."""
        action = self.action(task=task, max_retrials=max_retrials)
        action_msg = self.format_message(action.request, "ACTION")
        self.context += action_msg
        if action.argument:
            arg_msg = self.format_message(action.argument, "ARGUMENT")
            self.context += arg_msg
        os.system("cls" if os.name == "nt" else "clear")
        print(self.context)
        return action

    def execute_action(self, action: AgentAction, task: str) -> Optional[str]:
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
            elif action.request == "end_task":
                self.handle_end_task(task, action)
                return "end_task"
            else:
                raise ValueError(f"Unknown action request: {action.request}")

            if result is not None:
                obs_msg = self.format_message(str(result), "OBSERVATION")
                self.context += obs_msg
                if large_output:
                    self.large_observations.append((obs_msg))

        except Exception as e:
            error_msg = self.format_message(
                f"Error executing {action.request}: {str(e)}", "ERROR"
            )
            self.context += error_msg
        return None

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

                if section == max(toc.keys()):
                    end_index = len(book)
                else:
                    end_index = book.find(
                        f"{toc[section + 1]}\n==", start_index
                    )
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

            lines.insert(row - 1, code)

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

            lines[begin_row - 1 : end_row] = [code]

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
    def handle_end_task(self) -> str:
        """Handle the end task action."""
        end_task_msg = self.format_message("Task completed.", "END OF TASK")
        self.context += end_task_msg
        os.system("cls" if os.name == "nt" else "clear")
        print(self.context)

    # Formatting
    def format_message(self, text: str, action: str) -> str:
        """Format messages with indentation and color."""
        colored_action = self.color_text(f"{action} >> ", action)
        wrapped_text = textwrap.fill(text, width=100)
        return f"{colored_action}{wrapped_text}\n"

    def color_text(self, text: str, action: str) -> str:
        """Colorize text based on the action."""
        color_codes = {
            "REFLECTION": "\033[94m",
            "ACTION": "\033[92m",
            "OBSERVATION": "\033[93m",
            "ERROR": "\033[91m",
            "END OF TASK": "\033[96m",
            "ARGUMENT": "\033[90m",
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
        for obs_msg in self.large_observations:
            summary = self.summarize_text(obs_msg)
            summary_msg = self.format_message(summary, "SUMMARY")
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
                    "content": f"Please provide a concise summary of the following code using natural language:\n\n{text}",
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

    GPT_MODEL = "gpt-4o-mini"  # gpt-4o-2024-11-20
    OLLAMA_MODEL = "qwen2.5-coder:7b"

    SELECTED_MODEL = GPT_MODEL

    task = """
I have an implementation of the ray tracer algorithm in PyTorch. \
It is based on the book 'Ray Tracing In One Weekend', where the code is written in C++.

I want you to augment my code with additional features as described in the book 'Ray Tracing The Next Week', \
the second book of the series. You will be evaluated on the completeness and correctness of the implementation.

At the end of the task, I will personally review the code and evaluate your work. The nine scenes that are rendered \
in the book must be rendered correctly.

I recommend you to start the task by using the observe_repository tool to get a view of the current code. \
Then you need you should start implementing the code of the book's sections sequentially. \
Use the same style and structure as in the currently implemented code.
"""

    if SELECTED_MODEL == GPT_MODEL:
        agent = AgentReAct(
            model=SELECTED_MODEL,
            verbose=True,
        )
        agent.run_agent(task)
        agent.save_context_to_html("agent_context_gpt.html")

    elif SELECTED_MODEL == OLLAMA_MODEL:
        agent = AgentReAct(
            model=SELECTED_MODEL,
            verbose=True,
        )
        agent.run_agent(task)
        agent.save_context_to_html("agent_context_ollama.html")
