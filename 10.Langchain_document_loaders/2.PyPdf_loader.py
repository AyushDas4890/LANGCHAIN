from langchain_community.document_loaders import PyPDFLoader


###PyPDFLoader is a document loader in LangChain used to loader content from PDF files and convert each page into a Document
loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)