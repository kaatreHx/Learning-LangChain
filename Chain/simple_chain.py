from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = """
    Generate me some 3 fact about {topic}.
    
    """,
    input_variables=["topic"]
)

model_repo = "google/gemma-3-4b-it"

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

# res = chain.invoke({'topic': 'AI'})

chain.get_graph().print_ascii() # to get rough flow of chain