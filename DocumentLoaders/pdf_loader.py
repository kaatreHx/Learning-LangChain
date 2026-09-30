from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# loader = PyPDFLoader('DocumentLoaders/ai_book.pdf')

# docs = loader.load()

# Multiple files document loader
loader = DirectoryLoader(
    path='DocumentLoaders',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()
 
# for i in docs:
#     print(i)

print(next(docs))
print(next(docs))
