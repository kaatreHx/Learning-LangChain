from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch , RunnableSequence, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template = """
    Generate a report on {topic}.
    """,
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = """
    Summarize the report {text}
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

report_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, branch_chain)

res = final_chain.invoke({'topic': 'AI'})

print(res)

