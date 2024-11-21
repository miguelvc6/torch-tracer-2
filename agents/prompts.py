# System prompts for the language model
REFLECTION_SYSTEM_PROMPT = """GENERAL INSTRUCTIONS
Your task is to reflect on the question and context to decide how to solve it.
You must decide whether to use a tool, an assistant, or give the final answer if you have sufficient information.
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

AVAILABLE ASSISTANTS
- decomposition: {"Description": "Divides a complex question into simpler sub-parts and calls agents \
    to solve them recursively. Use only for complex questions", "Arguments": question - str}

AVAILABLE ACTION
- final_answer: {"Description": "Final answer for the user. Must answer the question asked.", "Arguments": "answer - str"}

RESPONSE FORMAT
REFLECTION >> <Fill>
"""

ACTION_SYSTEM_PROMPT_01 = """GENERAL INSTRUCTIONS
Your task is to answer questions using an SQL database and performing mathematical calculations.
If you already have enough information, you should provide a final answer.
You must decide whether to use a tool, an assistant, or give the final answer, and return a response following the response format.
Fill with null where no tool or assistant is required.

IMPORTANT:
- The response must be in valid JSON format.
- Ensure all text strings are properly escaped.
- Do not include line breaks within strings.
- If the argument is an SQL query or a mathematical expression, include it on a single line and in double quotes.

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
"""

ACTION_SYSTEM_PROMPT_DECOMPOSITION = """
AVAILABLE ASSISTANTS
- decomposition: {"Description: "Divides a complex question into simpler sub-parts and calls agents \
    to solve them recursively. Use only for complex questions", Arguments: question - str}
"""

ACTION_SYSTEM_PROMPT_02 = """
AVAILABLE ACTION
- final_answer: {"Description": "Final answer for the user. Must answer the question asked.", "Arguments": "answer - str"}

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

9. Final answer:
{
  "request": "final_answer",
  "argument": "Based on the repository content, the ray tracer implements motion blur in section 2."
}
"""
