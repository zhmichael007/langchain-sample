# pip install google-search-results
import os
os.environ["SERPAPI_API_KEY"]="2ff57bdfe29d2dfa2c4399215b978158ff558e2bc83b3f7e448de7e6da9ecb7c"

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import VertexAI

# The language model we're going to use to control the agent.
llm = VertexAI(temperature=0)

# The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Let's test it out!
agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
