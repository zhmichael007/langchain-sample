# Step 1. Load
print("***********************Load***********************")
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Step 2. Split
print("***********************Split***********************")
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# Step 3. Store
print("***********************Store***********************")
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=VertexAIEmbeddings())

# Step 4. Retrieve
print("***********************Retrieve***********************")
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print(len(docs))
# print (docs[0])
for doc in docs:
    print(doc.page_content[:300])

# Step 5. Generate
print("***********************Generate***********************")
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatVertexAI
llm = ChatVertexAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
result = qa_chain({"query": question})
print(result["result"])

f = open("resp", "w")
f.write(result["result"])
f.close()