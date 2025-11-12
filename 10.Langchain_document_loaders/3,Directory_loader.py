from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='10.langchain_document_loaders/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs_loader = loader.load()
docs = loader.lazy_load()

for document in docs:
    print(document.metadata)