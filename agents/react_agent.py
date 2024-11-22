import json
import logging
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
- edit_code: {
    "Description": "Edit code by replacing a specific code block with a new one using a diff-like approach", 
    "Arguments": {"file_path": str, "original_block": str, "new_block": str}
    }
- insert_code_block: {
    "Description": "Insert a new code block before or after a specific anchor block of code", 
    "Arguments": {"file_path": str, "anchor_block": str, "new_block": str, "position": str}
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
- edit_code: {
    "Description": "Edit code by replacing a specific code block with a new one using a diff-like approach", 
    "Arguments": {"file_path": str, "original_block": str, "new_block": str}
    }
- insert_code_block: {
    "Description": "Insert a new code block before or after a specific anchor block of code", 
    "Arguments": {"file_path": str, "anchor_block": str, "new_block": str, "position": str}
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

3. Using edit_code to modify existing code:
{
  "request": "edit_code",
  "argument": "{\"file_path\": \"src/utils.py\", \"original_block\": \"def old_method():\\n    pass\", \"new_block\": \"def old_method():\\n    return True\"}"
}

4. Using observe_repository without arguments:
{
  "request": "observe_repository",
  "argument": null
}

5. Using insert_code_block to add new code:
{
  "request": "insert_code_block",
  "argument": "{\"file_path\": \"src/camera.py\", \"anchor_block\": \"def get_focal_length(self):\\n    return self._focal_length\", \"new_block\": \"def new_method(self):\\n    pass\", \"position\": \"after\"}"
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
            elif action.request == "edit_code":
                args = json.loads(action.argument)
                result = self.edit_code(
                    args["file_path"],
                    args["original_block"],
                    args["new_block"],
                )
            elif action.request == "insert_code_block":
                args = json.loads(action.argument)
                result = self.insert_code_block(
                    args["file_path"],
                    args["anchor_block"],
                    args["new_block"],
                    args["position"],
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
        """
        Extracts content from a specified section of the book.
        """
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

            if section not in toc:
                raise ValueError(
                    f"Invalid section number. Must be between 1 and {len(toc)}"
                )

            with open("RayTracingTheNextWeek.html", encoding="utf-8") as f:
                book = f.read()
                search_term = f"{toc[section]}\n=="
                start_index = book.find(search_term)

                if start_index == -1:
                    return f"Section '{toc[section]}' not found in book."

                # For last section, read until end of file
                if section == max(toc.keys()):
                    return book[start_index:]

                # Otherwise find start of next section
                next_section = f"{toc[section + 1]}\n=="
                end_index = book.find(next_section, start_index)

                return (
                    book[start_index:end_index]
                    if end_index != -1
                    else book[start_index:]
                )

        except Exception as e:
            logging.error(f"Error observing book section {section}: {str(e)}")
            return None

    def run_main(self) -> Optional[str]:
        """
        Execute the main script and capture its output.
        """
        try:
            # Use context manager for better resource handling
            process = subprocess.run(
                [sys.executable, "src/main.py"],
                capture_output=True,
                text=True,
                check=True,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                },  # Ensure unbuffered output
            )
            return process.stdout
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running main script: {e.stderr}")
            return f"Error: {e.stderr}"
        except Exception as e:
            logging.error(f"Unexpected error running main script: {str(e)}")
            return None

    def observe_repository(self) -> Optional[str]:
        """
        Get concatenated content of all Python files in src/ directory.
        """
        try:
            src_path = "src/"
            if not os.path.exists(src_path):
                raise FileNotFoundError(
                    f"Source directory '{src_path}' not found"
                )

            concatenated_content = []

            for root, _, files in os.walk(src_path):
                # Sort files for consistent ordering
                for file in sorted(files):
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, src_path)

                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                separator = "=" * 100
                                concatenated_content.append(
                                    f"### {rel_path}\n{separator}\n{content}\n\n"
                                )
                        except Exception as e:
                            logging.error(
                                f"Error reading {file_path}: {str(e)}"
                            )
                            concatenated_content.append(
                                f"### {rel_path}\nError reading file: {str(e)}\n\n"
                            )

            return (
                "".join(concatenated_content)
                if concatenated_content
                else "No Python files found"
            )

        except Exception as e:
            logging.error(f"Error observing repository: {str(e)}")
            return None

    def edit_code(
        self, file_path: str, original_block: str, new_block: str
    ) -> Optional[str]:
        """
        Edit code by replacing a specific code block with a new one using a diff-like approach.
        The tool identifies the exact block to replace, ensuring precision in modifications.

        Args:
            file_path (str): Path to the target file
            original_block (str): The exact code block to be replaced (must match exactly)
            new_block (str): The new code block to insert

        Returns:
            Optional[str]: Success message, error message, or None if operation fails
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found")

            # Create backup
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)

            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Normalize line endings and whitespace in both blocks
                original_block = original_block.strip().replace("\r\n", "\n")
                new_block = new_block.strip().replace("\r\n", "\n")

                # Attempt to find the original block
                if original_block not in content:
                    # Try with normalized indentation
                    normalized_original = "\n".join(
                        line.strip() for line in original_block.split("\n")
                    )
                    normalized_content = "\n".join(
                        line.strip() for line in content.split("\n")
                    )

                    if normalized_original not in normalized_content:
                        raise ValueError(
                            "Original code block not found in file. "
                            "Please ensure the original block matches exactly."
                        )
                    else:
                        # Find the indentation of the original block
                        start_idx = content.find(
                            original_block.split("\n")[0].lstrip()
                        )
                        leading_whitespace = content[
                            start_idx
                            - content[start_idx::-1].find("\n") : start_idx
                        ]

                        # Apply original indentation to new block
                        new_block = "\n".join(
                            leading_whitespace + line if line.strip() else line
                            for line in new_block.split("\n")
                        )

                # Perform the replacement
                new_content = content.replace(original_block, new_block)

                # Ensure file ends with a newline
                if not new_content.endswith("\n"):
                    new_content += "\n"

                # Write the modified content
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                # Remove backup if successful
                os.remove(backup_path)

                return (
                    f"Code block successfully replaced in {file_path}\n"
                    f"Original block:\n{original_block}\n"
                    f"New block:\n{new_block}"
                )

            except Exception as e:
                # Restore backup on error
                shutil.move(backup_path, file_path)
                raise e

        except Exception as e:
            logging.error(f"Error editing code in {file_path}: {str(e)}")
            return f"Error: {str(e)}"

    def insert_code_block(
        self,
        file_path: str,
        anchor_block: str,
        new_block: str,
        position: str = "after",
    ) -> Optional[str]:
        """
        Insert a new code block before or after a specific anchor block of code.

        Args:
            file_path (str): Path to the target file
            anchor_block (str): The code block to use as reference point
            new_block (str): The new code block to insert
            position (str): Where to insert the new block ("before" or "after")

        Returns:
            Optional[str]: Success message, error message, or None if operation fails

        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found")

            if position not in ["before", "after"]:
                raise ValueError("Position must be either 'before' or 'after'")

            # Create backup
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)

            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Normalize line endings and whitespace
                anchor_block = anchor_block.strip().replace("\r\n", "\n")
                new_block = new_block.strip().replace("\r\n", "\n")

                # Find anchor block
                if anchor_block not in content:
                    raise ValueError(
                        "Anchor code block not found in file. "
                        "Please ensure the anchor block matches exactly."
                    )

                # Find proper indentation based on anchor block
                anchor_idx = content.find(anchor_block)
                if position == "after":
                    # For "after", find the next line's indentation
                    block_end = anchor_idx + len(anchor_block)
                    next_line_start = content.find("\n", block_end) + 1
                    if next_line_start > 0:
                        next_line_end = content.find("\n", next_line_start)
                        if next_line_end == -1:
                            next_line_end = len(content)
                        next_line = content[next_line_start:next_line_end]
                        indentation = len(next_line) - len(next_line.lstrip())
                else:
                    # For "before", use the anchor block's indentation
                    line_start = content.rfind("\n", 0, anchor_idx) + 1
                    indentation = anchor_idx - line_start

                # Apply indentation to new block
                new_block = "\n".join(
                    " " * indentation + line if line.strip() else line
                    for line in new_block.split("\n")
                )

                # Insert the new block
                if position == "after":
                    content = content.replace(
                        anchor_block, f"{anchor_block}\n{new_block}", 1
                    )
                else:
                    content = content.replace(
                        anchor_block, f"{new_block}\n{anchor_block}", 1
                    )

                # Ensure file ends with a newline
                if not content.endswith("\n"):
                    content += "\n"

                # Write the modified content
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                # Remove backup if successful
                os.remove(backup_path)

                return (
                    f"Code block successfully inserted {position} anchor in {file_path}\n"
                    f"Anchor block:\n{anchor_block}\n"
                    f"New block:\n{new_block}"
                )

            except Exception as e:
                # Restore backup on error
                shutil.move(backup_path, file_path)
                raise e

        except Exception as e:
            logging.error(
                f"Error inserting code block in {file_path}: {str(e)}"
            )
            return f"Error: {str(e)}"

    def rewrite_script(self, file_path: str, code: str) -> Optional[str]:
        """
        Rewrite entire content of a Python file.
        """
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Create backup if file exists
            if os.path.exists(file_path):
                backup_path = f"{file_path}.bak"
                shutil.copy2(file_path, backup_path)

            try:
                # Ensure code ends with newline
                if not code.endswith("\n"):
                    code += "\n"

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)

                if os.path.exists(f"{file_path}.bak"):
                    os.remove(f"{file_path}.bak")
                return f"Script {file_path} rewritten successfully"

            except Exception as e:
                # Restore backup if it exists
                if os.path.exists(f"{file_path}.bak"):
                    shutil.move(f"{file_path}.bak", file_path)
                raise e

        except Exception as e:
            logging.error(f"Error rewriting script: {str(e)}")
            return None

    def observe_single_script(self, script_name: str) -> Optional[str]:
        """
        Get content of a specific Python script from src/ directory.
        """
        try:
            file_path = f"src/{script_name}"
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Script '{script_name}' not found in src/ directory"
                )

            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        except Exception as e:
            logging.error(f"Error observing script {script_name}: {str(e)}")
            return None

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

    GPT_MODEL = "gpt-4o-2024-11-20"
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
