from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 

load_dotenv()

url = 'https://itechstore.com.np/product/macbook-pro-14-inch-m5-chip'
loader = WebBaseLoader(url)
docs = loader.load()

prompt = PromptTemplate(
    template='''
        Answer the following question \n {question} \n from the following text \n - \n {text}.
    ''',
    input_variables=['question','text']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'question':'What is the price of this macbook?','text':docs[0].page_content}))