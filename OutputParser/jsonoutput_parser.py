from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template = 'What is {topic}?\n{format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

chain = prompt | model | parser

res = chain.invoke({"topic": "LangChain"})

print(res)