from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import json

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

with open("StructuredOutput/json_schema.json", "r") as f:
    schema = json.load(f)

structured_llm = model.with_structured_output(schema, method="json_mode")

res = structured_llm.invoke(
    "Hi, There are 5 people and their names are Ram, Shyam, Hari, Sita and Gita. And their ages are 18, 19, 20, 21, 22."
)

print(res)