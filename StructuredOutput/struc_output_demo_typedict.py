from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    task="text-generation"
)   

class Person(TypedDict):
    name: str
    age: int
    gender: str
    eligible: Annotated[bool, "Whether the person is eligible for voting or not. Where minimum must be 19 years."]

model = ChatHuggingFace(llm=llm)

structured_llm = model.with_structured_output(Person)



res = structured_llm.invoke("Hi, My name is Ram Thapa. I am 18 years old and my gender is male.")

print(res)