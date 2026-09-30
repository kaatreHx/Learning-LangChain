from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='DocumentLoaders/students.csv')

docs = loader.load()

print(docs[1].page_content)