from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2", temperature=0)

sql_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an Oracle SQL expert.

Table: employees(emp_id, name, department, salary, hire_date)

Return ONLY SQL.

Question: {question}

SQL:
"""
)

def generate_sql(question):
    chain = sql_prompt | llm
    return chain.invoke({"question": question}).content.strip()