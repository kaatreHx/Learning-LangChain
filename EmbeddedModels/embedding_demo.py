from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

res = model.embed_query("what is the capital of india")
print(str(res))