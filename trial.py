import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "faeadaef64b2e06390567b9a23550a32b5912af3" # serper.dev API key

# You can choose to use a local model through Ollama for example. See https://docs.crewai.com/how-to/LLM-Connections/ for more information.

# os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
# os.environ["OPENAI_MODEL_NAME"] ='openhermes'  # Adjust based on available model
# os.environ["OPENAI_API_KEY"] ='sk-111111111111111111111111111111111111111111111111'

# You can pass an optional llm attribute specifying what model you wanna use.
# It can be a local model through Ollama / LM Studio or a remote
# model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)
#
# import os
os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
#
# OR
#
# from langchain_openai import ChatOpenAI

search_tool = SerperDevTool()

code_body = """"def myPow(x, n):
        '''
        Implement pow(x, n), which calculates x raised to the power n (i.e., xn).

        :type x: float
        :type n: int
        :rtype: float
        '''
        if n == 0:
            return 1
        
        if n < 0:
            return 1/myPow(x, -n)
        
        if n%2 == 0:
            return myPow(x*x, n/2)
        
        return x * myPow(x, n-1)"""

# Define your agents with roles and goals
code_understander = Agent(
  role='Software Engineer',
  goal='Understand what a function of code is doing',
  backstory="""You are a world renowned software engineer who's capable of reading any code function and figuring out its function. 
  You first understand semantically what the code does and afterwards, understand how each potential code flow in the function could be triggered.
  """,
  verbose=True,
  allow_delegation=False,
  # You can pass an optional llm attribute specifying what model you wanna use.
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
  tools=[search_tool]
)
edge_case_understander = Agent(
  role='Software Engineer',
  goal='Think of possible edge cases in a function',
  backstory="""You are a world renowned software engineer. Given any body of code and an idea of what it does,
  you are able to brainstorm (at a high level) possible edge cases to the function. This includes edge cases regarding
  logic but also making sure that your cases cover every possible code flow in the function.""",
  verbose=True,
  allow_delegation=True
)

input_maker = Agent(
  role='Software Engineer',
  goal='Think of possible inputs for test cases for a function',
  backstory="""You are a world renowned software engineer. Given code for a function as well as a high level list of test cases, 
  you are the best at creating inputs (not outputs) that match every test case given to you.""",
  verbose=True,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description=f"""Use the following as code: {code_body}. Read through it and understand very specifically 
  what the function does. Then, look at the conditionals and understand the flow of logic throughout the code. See how every possible path is reached.""",
  expected_output="Summary of the code in a paragraph. For each code flow, write in bullet points how to reach that flow.",
  agent=code_understander,
)

task2 = Task(
  description="""Ask your coworker for its understanding of the function. Based on the summary and the description of how each path is reached, 
  think of possible edge cases that might not seem obvious at first but could possibly be handled incorrectly by the code.""",
  expected_output="List of edge cases and why you think the code might incorrectly handle them.",
  agent=edge_case_understander
)

task3 = Task(
  description="""Ask your coworkers for the function summary, code flow, and edge case list. Brainstorm a list of possible test cases to test the function but for each test case, only provide the input (not the output as well)""",
  expected_output="List of inputs for each test case as well as a brief description of what the test case tests.",
  agent=edge_case_understander
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[code_understander, edge_case_understander, input_maker],
  tasks=[task1, task2, task3],
  verbose=2, # You can set it to 1 or 2 to different logging levels
  cache = True
)



# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)