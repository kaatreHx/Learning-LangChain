from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#Chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are helpful customer suppport agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{question}'),
])

chat_history = []
#Load chat history
with open('Prompts/Chat.txt') as f:
    chat_history.extend(f.readlines())

#Create prompt

prompt = chat_template.invoke({'chat_history': chat_history, 'question': 'where is my refund?'})

print(prompt)
