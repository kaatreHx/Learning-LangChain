from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive","Negative"] = Field(description="Sentiment of the feedback")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="""
   Classify the sentiment of the following feedback text into positive or negative \n {feedback}\n {format_instructions}  """,
    input_variables=["feedback"],
    partial_variables={"format_instructions":pydantic_parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="""
    Write an appropriate response to this positive feedback \n {feedback}    """,
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="""
    Write an appropriate response to this negative feedback \n {feedback}    """,
    input_variables=["feedback"]
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

classify_chain = prompt1 | model | pydantic_parser

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Invalid feedback")
)

chain = classify_chain | branch_chain
print(chain.invoke({'feedback':'It was bad phone'}))