from crewai import Agent, Task
from crewai_tools import SerperDevTool


def create_agents():
    code_understander = Agent(
        role='Software Engineer',
        goal='Understand what a function of code is doing',
        backstory="""You are a world renowned software engineer who's capable of reading any code function and figuring out its function.
        You first understand semantically what the code does and afterwards, understand how each potential code flow in the function could be triggered.
        To do so, you never consult Google. SEARCHING THE INTERNET IS BANNED!!! NO GOOGLE SEARCHING!
        """,
        verbose=True,
        allow_delegation=False,
    )

    edge_case_understander = Agent(
        role='Software Engineer',
        goal='Think of possible edge cases in a function',
        backstory="""You are a world renowned software engineer. Given any body of code and an idea of what it does,
        you are able to brainstorm (at a high level) possible edge cases to the function. This includes edge cases regarding
        logic but also making sure that your cases cover every possible code flow in the function.
        To do so, you never consult Google. SEARCHING THE INTERNET IS BANNED!!! NO GOOGLE SEARCHING!""",
        verbose=True,
        allow_delegation=False
    )

    input_maker = Agent(
        role='Software Engineer',
        goal='Think of possible inputs for test cases for a function',
        backstory="""You are a world renowned software engineer. Given code for a function as well as a high level list of test cases,
        you are the best at creating inputs (not outputs) that match every test case given to you.
        To do so, you never consult Google. SEARCHING THE INTERNET IS BANNED!!! NO GOOGLE SEARCHING!""",
        verbose=True,
        allow_delegation=True
    )

    oracle = Agent(
        role='Question Answer',
        goal='Make sure that each input for each test case actually satisfies the desired condition.',
        backstory="""You are extremely inquisitive and love to ask questions. So, you've been hired to the crew to ensure that each input/test case being created
        actually tests what we expect it to. Using the list of inputs just created, go through each input and test case and make sure that the input actually actually
        matches what the test case's description is.""",
        verbose=True,
        allow_delegation=True
    )

    code_runner  = Agent(
        role='Code Runner',
        goal='Given all of the test cases with inputs and outputs, make sure that our original code function produces the right answer for each input.',
        backstory="""You love running code. You have to run the code against each test case.""",
        verbose=True,
        allow_delegation=False
    )

    output_maker = Agent(
        role='Problem Solver',
        goal='Given a list of test cases with their inputs as well as an understanding of what the function is supposed to do, give the output for each input.',
        backstory="""You love solving problems.""",
        verbose=True,
        allow_delegation=True
    )

    code_fixer = Agent(
        role='Code Writer',
        goal='If the code failed a test case, fix the code.',
        backstory="""You are the world's most advanced coder and can solve any coding problem.""",
        verbose=True,
        allow_delegation=True
    )

    return {
        'code_understander': code_understander,
        'edge_case_understander': edge_case_understander,
        'input_maker': input_maker,
        'oracle': oracle,
        'code_runner': code_runner,
        'output_maker': output_maker,
        'code_fixer': code_fixer,
    }

def build_test_generation_task_list(agents, code_body):
    task1 = Task(
        description=f"""Use the following as code: {code_body}. Read through it and understand very specifically
        what the function does. Then, look at the conditionals and understand the flow of logic throughout the code. See how every possible path is reached.""",
        expected_output="Summary of the code in a paragraph. For each code flow, write in bullet points how to reach that flow.",
        agent=agents['code_understander'],
    )

    task2 = Task(
        description="""Based on the summary of the function and the description of how each path is reached,
        think of possible edge cases that might not seem obvious at first but could possibly be handled incorrectly by the code.""",
        expected_output="List of edge cases and why you think the code might incorrectly handle them.",
        agent=agents['edge_case_understander']
    )

    task3 = Task(
        description="""Based on the function summary, code flow, and edge case list, brainstorm a list of possible test cases to test the function but for each test case, only provide the input (not the output as well)""",
        expected_output="List of inputs for each test case as well as a brief description of what the test case tests.",
        agent=agents['input_maker']
    )

    task4 = Task(
        description="""You are extremely inquisitive and love to ask questions. So, you've been hired to the crew to ensure that each input/test case being created
        actually tests what we expect it to. Using the list of inputs just created, go through each input and test case and make sure that the input actually actually
        matches what the test case's description is.""",
        expected_output="List of inputs for each test case as well as a brief description of what the test case tests.",
        agent=agents['oracle']
    )

    task5 = Task(
        description=f"""Given a list of test cases with their inputs as well as an understanding of what the function is supposed to do, give the output for each input. Do this without running the code itself.
        Instead, use problem solving abilities to solve the problem for yourself to get the output. As a reminder, the original function is given here: {code_body}""",
        expected_output="List of test cases with input, output, and what is being tested with that test case.",
        agent=agents['output_maker']
    )

    task6 = Task(
        description=f"""Given all of the test cases with inputs and outputs, chceck if our original code function produces the right answer for each input. To do so, run the original code in a IDE and see what answer is given
        for each input. It is ok if the code output is incorrect. As a reminder, here's the code: {code_body}.

        There will be times when you are summoned by the code writer. When that happens, you will be told to rerun new code against the test cases. Use the test cases you created previously and run the new code that is being
        provided to you. Then, return back the list of test cases and whether they worked or did not work. Additionally, return back to the code writer the code that it just sent you so that it is able to remember it.""",
        expected_output="List of test cases that worked and test cases that did not work",
        agent=agents['code_runner']
    )

    task7 = Task(
        description=f"""You are given the list of test cases and whether they succeeded or failed. For every failed test case, see why it failed and adjust the code accordingly (no delegation allowed here). After you diagnose the error,
        fix the code so that the error is no longer there. Then, go to the code runner and have them rerun the code to ensure that all the test cases now work. Keep repeating this cycle until all of the test cases work or until 20 repetitions have occurred.
        Every time you prompt the code runner, give it the code as part of your query so it remembers what code to run.

        Once all test cases come become correct, you're done.""",
        expected_output=f"""List of test cases as well as the new code. Put those in a JSON file of the form:
        {{
            test_cases: [the test_cases],
            code: [the updated code]
        }}. Before submitting the code, double check to make sure it is proper JSON and that if I run json.loads on the output, I won't get an error.

        In [the updated code], make sure to put the entire code body even if no updates were made. The no updates code is found here: {code_body}. However, make sure it is formatted
        as one big string.""",
        agent=agents['code_fixer']
    )

    return [task1, task2, task3, task4, task5, task6, task7]

def create_python_from_tests(test_results, code_body, agents):
    test_cases = test_results['test_cases']
    task1 = Task(
        description=f"""
        Given the test cases and the code body, write a function in Python that satisfies the test cases. The test cases are as follows:
        {test_cases}
        The code body is as follows:
        {code_body}""",
        expected_output="A string that contains the Python code that satisfies the test cases. Make sure that the code is properly formatted and that it is a string.",
        agent=agents['code_runner'],
    )

    return [task1]
