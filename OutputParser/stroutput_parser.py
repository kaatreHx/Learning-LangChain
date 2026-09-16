from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'What is {topic}?',
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

res = chain.invoke({"topic": "LangChain"})

print(res)