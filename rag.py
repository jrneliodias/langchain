import os
import dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
from langchain.prompts import PromptTemplate
dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


# Helper function to clean metadata by replacing None values with empty strings
def clean_metadata(metadata):
    return {k: (v if v is not None else '') for k, v in metadata.items()}


model = ChatOpenAI(
    model="gpt-4",
    temperature=0.9)

conn = sqlite3.connect('database.db')

df_food = pd.read_sql_query("SELECT * FROM places_to_eat", conn)
df_visit = pd.read_sql_query("SELECT * FROM places_to_visit", conn)

conn.close()


# Combine all columns into a single 'text' column
df_food['text'] = df_food.apply(lambda row: ' '.join([str(value)
                                                      for value in row if value is not None]), axis=1)
df_visit['text'] = df_visit.apply(lambda row: ' '.join([str(value)
                                                        for value in row if value is not None]), axis=1)

documents_food = [
    Document(page_content=row['text'], metadata=clean_metadata(row.to_dict()))
    for _, row in df_food.iterrows()
]

documents_visit = [
    Document(page_content=row['text'], metadata=clean_metadata(row.to_dict()))
    for _, row in df_visit.iterrows()
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

chunks_food = text_splitter.split_documents(documents_food)
chunks_visit = text_splitter.split_documents(documents_visit)

embedding = OpenAIEmbeddings(
)
persisted_directory = 'vector_store_db'

chunks = chunks_food + chunks_visit

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name='amazon_places',
    persist_directory=persisted_directory,
)

retriver = vector_store.as_retriever()

prompt_knowledge = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    Be a place to amazon guider for places to eat, visit, service and rest. Answer in the same language the following question.
    use only this knowledge to answer and don't use external sources from this context. If you don't know the answer, just say that you don't know in
    the same language of the question and try to help the user ask question about you and your knowledge. 
    If you have the answer, use the maximum amount of information you can with bullet points and list the same language of the question. If your data is another language,
    translate you data to the language of the question. All your answer must be in the same language of the question and you have to translate it.
    Context: {context}
    Question: {question}
    """

)


rag_chain = (
    {
        'context': retriver,
        'question': RunnablePassthrough(),
    }
    | prompt_knowledge
    | model
    | StrOutputParser()
)

question = 'Onde posso comer no Carreiro?'

response = rag_chain.invoke(question)

print(response)
