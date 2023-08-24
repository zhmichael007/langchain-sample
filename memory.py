from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatVertexAI
from langchain.memory import ConversationBufferMemory

import time

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context. If the AI does not know the answer to a "
        "question, it truthfully says it does not know."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatVertexAI(temperature=0, top_p=0.8,max_output_tokens=1024)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

res = conversation.predict(input="Hi there!")
print("User input: "+"Hi there!")
print("LLM output: "+res)
time.sleep(2)

res = conversation.predict(input="My name is Michael.")
print("User input: "+"My name is Michael.")
print("LLM output: "+res)
time.sleep(2)

res = conversation.predict(input="Do you know what's my name?")
print("User input: "+"Who am I?")
print("LLM output: "+res)
time.sleep(2)

load=memory.load_memory_variables({})

print(load)