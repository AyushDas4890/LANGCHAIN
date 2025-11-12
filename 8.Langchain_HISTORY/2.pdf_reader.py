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

#Create a retriever (fetches relevent documents)8.Langchain_HISTORY/2.pdf_reader.py
retriever = vectorstore.as_retriever()

# Manually Retrieve relevant documents
query = "What are the takeaways from the document?"
relevant_docs = retriever.get_relevant_documents(query)

# combine Retrieved docs into a single Prompt
retrieved_text = "\n".join([doc.page_content for doc in relevant_docs])

# Initialize the LLM
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Manually pass Retrived text to LLM
prompt = f"Based on the following document excerpts, answer the question: {query}\n\n{retrieved_text}"
answer = llm.predict(prompt)

# Print the Answer
print("Answer:", answer)

