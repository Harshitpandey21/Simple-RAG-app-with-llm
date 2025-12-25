from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader('sample.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

query = "provide me the summary for this document"
retrieved_docs = retriever.invoke(query)

retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

llm = ChatOpenAI(model = "gpt-4o", temperature=0.7)

prompt = f"Based on the following text, answer the question:{query}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

print("Answer:", answer.content)