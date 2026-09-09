from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

model = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

document = [
    "Ram is Doctor. He is 25 years old",
    "Laxman is engineer. He is 26 years old",
    "Sita is teacher. She is 25 years old",
    "Hanuman is Docter. He is 26 years old"
]

query = "Whhich doctor is more younger?"
query_embedding = model.embed_query(query)

doc_embeddings = model.embed_documents(document)

res = cosine_similarity(
    [query_embedding], doc_embeddings
)[0]

index, score = sorted(list(enumerate(res)), key=lambda x:x[1])[-1]

print(f"The most similar document is: {document[index]} with a similarity score of {score}")