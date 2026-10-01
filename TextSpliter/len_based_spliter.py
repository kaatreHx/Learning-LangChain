from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('DocumentLoaders/ai_book.pdf')
docs = loader.load()

# """
#     this split the document based on the length of the document.
#     chunk_size : the size of the chunk
#     chunk_overlap : the overlap between the chunks
# """

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

splitedDocs = splitter.split_documents(docs)

print(splitedDocs[0].page_content)

#. While chunking the docs are split based on below conditions
# First it split the paragraph then sentence then space after that character