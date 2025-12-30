import ollama
import json
import re

TASK_SPLITTER_SYSTEM_PROMPT = """
You are a task decomposition assistant.

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
    "task": "Make a system that can remind me to go running every day."
}

output_value = run_task_splitter(input_value["task"])
print(output_value["subtasks"])
