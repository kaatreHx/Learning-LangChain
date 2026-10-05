from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="my_documents", #kind a table name
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

vector_store.add_texts([
    "Python is a programming language",
    "Django is a Python web framework",
    "LangChain is used for building LLM applications"
])

#Get all emebedded and store vector
# results = vector_store.get() 
# print(results)

vector_store.add_texts(
    ["Python is a popular programming language"],
    ids=["python-1"]
)

#This returns list and give the nearest matching vectors as per query
# results = vector_store.similarity_search(
#     "What is popular language?",
#     k = 2
# )

# for data in results:
#     print(data.id)

vector_store.update_documents(
    ids=["python-1"],
    documents=[
        Document(
            page_content="Python is a popular interpreted language",
            metadata={"updated":True}
        )
    ]
)

vector_store.delete(ids=["python-1"]) #delete the vectors by ids
print(vector_store.get(ids=["python-1"]))

