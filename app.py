import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, Process, LLM
from crewai.tools import tool

load_dotenv()
# to create object of LLM or configuration of LLM
llm = LLM(
    model="gemini/gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Using custom tool
@tool("calculator")
def calculator(query:str) -> str:
    """A simple calculator tool that evaluates math expressions."""

    try:
        result = eval(query)
        return f"The result of {query} is {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# agent - 1
satya = Agent(
    role="Satya - Researcher",
    goal="Perform research and compute results using tools.",
    backstory="Handles calculations and prepares data for reporting.",
    tools=[calculator], 
    verbose=True,
    llm = llm
)

# agent - 2
narayana = Agent(
    role="Narayana - Writer",
    goal="Summarize the research into a readable article.",
    backstory="Takes calculation results and writes summaries for users",
    verbose=True,
    llm = llm
)

# Assigning Tasks
task1 = Task(
    description="Calculate '12 * (5 + 3)' using the calculator tool.",
    agent=satya,
    expected_output="The numeric result of the expression."
)

task2 = Task(
    description="Summarize the researcher's calculation in a short article.",
    agent=narayana,
    expected_output="A summary article including the calculation result.",
    context=[task1],
    output_file="final_report.md"
)

# the object of crew
crew = Crew(
    agents=[satya, narayana], tasks=[task1, task2], process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("--- Gemini Agentic Workflow Started ---")
    result = crew.kickoff()
    
    print("\n\n========================")
    print("## FINAL REPORT ##")
    print("========================\n")
    print(result)