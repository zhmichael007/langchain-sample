import os
import getpass
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import FAISS

loader = PyPDFLoader("./user_guide.pdf")
pages = loader.load_and_split()
# print(pages[1].page_content)

question="How to take recording"
faiss_index = FAISS.from_documents(pages, VertexAIEmbeddings())
docs = faiss_index.similarity_search(question, k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

# Bullets with VertexAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatVertexAI
llm = ChatVertexAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=faiss_index.as_retriever())
result = qa_chain({"query": question})
print(result["result"])

f = open("resp", "w")
f.write(result["result"])
f.close()