# set the project to the demo project for Generative AI Fishfood
# gcloud config set project cloud-llm-preview1

from langchain.chat_models import ChatVertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Chinese."
    ),
    HumanMessage(
        content="Translate this sentence from English to Chinese. I love programming."
    ),
]

chat = ChatVertexAI(temperature=0.2, max_output_tokens=256,
                    top_p=0.8, top_k=40)

res = chat(messages)
print(res.content)

# ****************************************************************************************
template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# get a chat completion from the formatted messages
res = chat(
    chat_prompt.format_prompt(
        input_language="English", output_language="French", text="I love programming."
    ).to_messages()
)
print(res.content)

# ****************************************************************************************
chat = ChatVertexAI(model_name="chat-bison")
messages = [
    HumanMessage(
        content="How do I create a python function to identify all prime numbers?"
        # content="Give me a Java sample code of google cloud, generate an access token from Service Account key file, and generate a down scopes access token from this service acount access token with restrict a credential's Cloud Storage permissions according to this link: https://cloud.google.com/iam/docs/downscoping-short-lived-credentials. "
    )
]
res = chat(messages)
print(res.content)
f = open("prime.md", "w")
f.write(res.content)
f.close()
