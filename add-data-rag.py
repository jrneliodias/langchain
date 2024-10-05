import os
import dotenv
import pandas as pd
import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document

dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Helper function to clean metadata by replacing None values with empty strings


def clean_metadata(metadata):
    return {k: (v if v is not None else '') for k, v in metadata.items()}

# Function to fetch data from a specific table


def fetch_table_data(table_name):
    conn = sqlite3.connect('database.db')
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to convert table data into documents


def table_to_documents(df):
    # Combine all columns into a single 'text' column
    df['text'] = df.apply(lambda row: ' '.join([str(value)
                                                for value in row if value is not None]), axis=1)

    documents = [
        Document(page_content=row['text'],
                 metadata=clean_metadata(row.to_dict()))
        for _, row in df.iterrows()
    ]
    return documents

# Function to split documents into chunks


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
    )
    return text_splitter.split_documents(documents)

# Function to add documents to the vector store


def add_to_vector_store(chunks, collection_name, persisted_directory='vector_store_db'):
    embedding = OpenAIEmbeddings()
    vector_store = Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
        persist_directory=persisted_directory,
    )
    vector_store.add_documents(chunks)
    return vector_store

# Main function to process a table and add it to the vector store


def process_tables(table_names, collection_name):
    for table_name in table_names:
        try:
            # Fetch the table data
            df = fetch_table_data(table_name)

            # Convert the table into documents
            documents = table_to_documents(df)

            # Split documents into chunks
            chunks = split_documents(documents)

            # Add documents to the vector store
            vector_store = add_to_vector_store(chunks, collection_name)

            print(f"Added {table_name} to the vector store")

        except Exception as e:
            print(f"Error processing {table_name}: {e}")


# Example usage
if __name__ == "__main__":
    # Replace with your specific table name
    table_names = ["places_to_eat",
                   "places_to_visit",
                   "places_to_service",
                   "places_to_rest"]
    collection_name = "amazon_places"  # Replace with your specific collection name
    process_tables(table_names, collection_name)
