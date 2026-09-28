from langchain_community.document_loaders import TextLoader

loader = TextLoader("DocumentLoaders/Info.txt", encoding="utf-8")
doc = loader.load()

print(doc[0].metadata)


