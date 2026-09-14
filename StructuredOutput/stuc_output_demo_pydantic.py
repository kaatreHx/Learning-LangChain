from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

class PersonData(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(description="Age of person")

model = ChatHuggingFace(llm=llm)

# Fix: Force JSON mode extraction instead of function calling
structured_llm = model.with_structured_output(PersonData, method="json_mode")

res = structured_llm.invoke(
    "Hi, There are 5 people and their names are Ram, Shyam, Hari, Sita and Gita. And their ages are 18, 19, 20, 21, 22."
)

print(res)