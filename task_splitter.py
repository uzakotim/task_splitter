import ollama
import json
import re

TASK_SPLITTER_SYSTEM_PROMPT = """
You are a task decomposition assistant and software engineering expert.
Make a deep analysis of the task and return step-by-step instructions to implement it.

Your job is to take a single task and:
1. Explain the strategy used to split the task
2. Return a list of subtasks



Output MUST be valid JSON with the following schema:

{
  "strategy": "string",
  "subtasks": ["string", "string", ...]
}

Do NOT include markdown, explanations, or extra text.
Return JSON only.
"""

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found. Raw output:\n{text}")
    return json.loads(match.group())

def run_task_splitter(task: str, model: str = "gemma2:2b") -> dict:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": TASK_SPLITTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task}"},
        ],
        options={
            "temperature": 0.2,
            "num_predict": 300,
        },
    )

    content = response["message"]["content"]
    return extract_json(content)

input_value = {
    "task": "How to make a dummy pod controlled by buttons in AR for iOS"
}

print(input_value["task"])
print()
output_value = run_task_splitter(input_value["task"])
# print(output_value) each subtask separately

for subtask in output_value["subtasks"]:
    print(subtask)