from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='name', description='Name of the person'),
    ResponseSchema(name='age', description='Age of the person'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    template = 'Fictional character name and age. where the name is {character}?\n{format_instruction}',
    input_variables = ['character'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

chain = prompt | model | parser

res = chain.invoke({"character": "Ironman"})

print(res)