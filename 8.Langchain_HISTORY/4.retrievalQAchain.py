from langchain.document_loaders import TextLoader
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import FAISS
from langchain.llms import OpenAI   

# load the document
loader = TextLoader('docs.txt') # Ensure docs.txt exist
documents = loader.load()

# split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# convert text into embeddings & store in FAISS vector store
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

#Create a retriever (fetches relevent documents)
retriever = vectorstore.as_retriever()


#Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Create RetrievalQAChain
qa_chain = RetrievalQAChain.from_llm(llm, retriever=retriever)

#Ask a questionwww
query = "What are the takeaways from the document?"
answer = qa_chain.run(query)


print("Answer:", answer)