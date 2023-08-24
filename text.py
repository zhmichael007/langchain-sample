from langchain.llms import VertexAI
#from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
template = """Question: {question} Answer: Please reply in Chinese."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = VertexAI(temperature=0.2)
#llm = OpenAI(temperature=0.9)
question = "Give me a introduction about Stanford University."
llm_chain = LLMChain(prompt=prompt, llm=llm)
res = llm_chain.run(question)
print(res)