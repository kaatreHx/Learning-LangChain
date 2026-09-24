from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template = """
    Generate a detailed report on {topic}.
    """,
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = """
    Generate a 5 pointer summary from the following text \n {text}
    """,
    input_variables=["text"]
)

model_repo = "google/gemma-3-4b-it"

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

res = chain.invoke({'topic': 'AI'})

print(res)

# chain.get_graph().print_ascii()
