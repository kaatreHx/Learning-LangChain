from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template = """
    Generate short and simple notes from the following text \n {text}.
    """,
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = """
    Generate 5 short question from the following text \n {text}
    """,
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template = """
    Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}
    """,
    input_variables=["notes", "quiz"]
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
)  

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.1
)

# llm3 = HuggingFaceEndpoint(
#     repo_id="prism-ml/Ternary-Bonsai-2-27B-gguf",
#     task="text-generation",
#     temperature=0.1
# )

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm2)
# model3 = ChatHuggingFace(llm=llm3)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """
Introduction and Overview

Django is a high-level, open-source Python web framework used to build secure, scalable, and maintainable web applications. It follows the MVT (Model-View-Template) architecture and provides built-in features that make web development faster and easier.

Django is designed to simplify the development process by providing tools for handling databases, user authentication, URL routing, forms, and security.

Key Features of Django
Easy to Learn: Django uses Python, which has simple and readable syntax.
MVT Architecture: Separates application logic, data handling, and user interface.
Built-in Admin Panel: Provides an admin interface for managing database records.
ORM (Object-Relational Mapping): Allows developers to interact with databases using Python instead of writing SQL queries.
Security: Includes protection against common vulnerabilities such as CSRF, SQL injection, and XSS.
Scalability: Suitable for small projects as well as large web applications.
URL Routing: Makes it easy to map URLs to specific views.
Why Use Django?

Django helps developers build web applications quickly by providing reusable components and built-in functionality. It is commonly used for blogs, e-commerce platforms, social media applications, APIs, and enterprise applications.

Example: Django can be used to create a real estate management system where users can register, log in, add properties, and search for available properties.

Conclusion

Django is a powerful Python web framework that simplifies web development through its built-in tools, security features, and structured architecture. It is a good choice for developers who want to create reliable and maintainable web applications.
"""

res = chain.invoke({'text': text})

print(res)
