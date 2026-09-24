from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template = """
    Generate a tweet on {topic}.
    """,
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = """
    Generate a Linkdin post about {topic}
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

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkdin": RunnableSequence(prompt2, model , parser)
})

res = parallel_chain.invoke({'topic': 'AI'})

print(res)

# chain.get_graph().print_ascii()
