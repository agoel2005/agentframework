import json
import os
from datetime import datetime

# from openai import OpenAI
from crewai import Crew

from constants import OPENAI_LLM_MODEL
from crew import (build_test_generation_task_list, create_agents,
                  create_python_from_tests)


def save_results(result, file_path):
    STORE_FILE_NAME = 'storage.txt'
    JSON_FILE_NAME = 'storage_maxpoints.json'

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    result_str = result
    ans = ""
    found = False

    for res in result_str.splitlines():
        if "```json" in res:
            found = True
            continue
        elif "```" in res:
            found = False
            continue

        if found:
            ans += res + "\n"

    ans = ans.rstrip("\n")

    with open(f'{file_path}/{STORE_FILE_NAME}', "w") as f:
        f.write(ans)
        f.close()

    print(ans)
    format = json.loads(ans)

    with open(f'{file_path}/{JSON_FILE_NAME}', "w") as f:
        json.dump(format, f)
        f.close()

def write_python_file(file_path, code_body):
    with open(f'{file_path}/tests.py', 'w') as f:
        f.write(code_body)
    return

def main():
    os.environ['OPENAI_MODEL_NAME'] = OPENAI_LLM_MODEL
    FILE_NAME = 'a0001twosum.py'
    FILE_PATH = f'./solutions/{FILE_NAME}'

    time = datetime.now()
    SAVE_FILE_PATH = f'./results/{time}_{FILE_NAME}'

    code_body = None

    with open(FILE_PATH, 'r') as f:
        code_body = f.read()
    assert(code_body is not None)

    agents_obj = create_agents()
    agents_list = list(agents_obj.values())
    test_creation_tasks = build_test_generation_task_list(agents_obj, code_body)

    crew = Crew(
        agents=agents_list,
        tasks=test_creation_tasks,
        verbose=2, # You can set it to 1 or 2 to different logging levels
        cache = True,
        memory=False,
    )

    result = crew.kickoff()

    print("######################")
    print(result)
    save_results(result, SAVE_FILE_PATH)

    test_code_tasks = create_python_from_tests(agents_obj, code_body)

    crew = Crew(
        agents=[agents_obj['code_runner']],
        tasks=test_code_tasks,
        verbose=2, # You can set it to 1 or 2 to different logging levels
        cache = True,
        memory=False,
    )

    result = crew.kickoff()

    print("######################")
    print(result)

    write_python_file(SAVE_FILE_PATH, result)
    return

if __name__ == '__main__':
    main()
