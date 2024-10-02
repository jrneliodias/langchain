
import os
import dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DataFrameLoader, UnstructuredExcelLoader
import pandas as pd
from langchain_community.utilities.sql_database import SQLDatabase
import sqlite3

dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

set_llm_cache(SQLiteCache(database_path="langchain.db"))

# model = OpenAI()
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9)

conn = sqlite3.connect('database.db')

df = pd.read_sql_query("SELECT * FROM places_to_eat", conn)

conn.close()


# Combine all columns into a single 'text' column
df['text'] = df.apply(lambda row: ' '.join([str(value)
                      for value in row]), axis=1)


loader = DataFrameLoader(df, page_content_column='text')

documents = loader.load()


prompt_knowledge = PromptTemplate(
    input_variables=["question, context"],
    template="""
    Be a place to eat guider. Answer in the same language the following question
    use only this knowledge to answer and don't use external sources from this context.
    Context: {context}
    Question: {question}
    """

)

classification_chain = (
    PromptTemplate.from_template(
        """ 
        Classify the question in one of the following categories:
        - food
        - sleep
        - visit
        - services
        question: {question}
    """
    )
    | model
    | StrOutputParser()
)


eat_chain = (
    prompt_knowledge
    | model
    | StrOutputParser()
)


def route(classification):

    if "food" in classification:
        return eat_chain


question = 'What city you have restaurant data?'

classification = classification_chain.invoke({'question': question})
print(classification)


response = eat_chain.invoke({"question": question, "context": documents})

print(response)
