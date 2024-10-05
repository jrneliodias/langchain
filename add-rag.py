import os
import dotenv
import pandas as pd
import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.schema import Document
dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


# Helper function to clean metadata by replacing None values with empty strings
def clean_metadata(metadata):
    return {k: (v if v is not None else '') for k, v in metadata.items()}


conn = sqlite3.connect('database.db')

df = pd.read_sql_query("SELECT * FROM places_to_eat", conn)


conn.close()


# Combine all columns into a single 'text' column
df['text'] = df.apply(lambda row: ' '.join([str(value)
                                            for value in row if value is not None]), axis=1)


documents = [
    Document(page_content=row['text'], metadata=clean_metadata(row.to_dict()))
    for _, row in df.iterrows()
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(documents)


embedding = OpenAIEmbeddings(
)

persisted_directory = 'vector_store_db'

vector_store = Chroma(
    documents=chunks,
    embedding_function=embedding,
    collection_name='amazon_places',
    persist_directory=persisted_directory,
)

vector_store.add_documents(chunks)
