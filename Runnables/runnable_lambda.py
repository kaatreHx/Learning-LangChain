from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableSequence, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template = """
    Generate a joke on {topic}.
    """,
    input_variables=["topic"]
)

def wordCount(text):
    return len(text.split())

model_repo = "google/gemma-3-4b-it"

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model, parser)
    
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(), 
    "word_count": RunnableLambda(wordCount)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

res = final_chain.invoke({'topic': 'AI'})

print(res)

# chain.get_graph().print_ascii()
