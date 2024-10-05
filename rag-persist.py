import os
import dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import time
from langchain.globals import set_debug
dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

start_time = time.time()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.8)

set_debug(True)

embedding = OpenAIEmbeddings(
)
persisted_directory = 'vector_store_db'


vector_store = Chroma(
    embedding_function=embedding,
    collection_name='amazon_places',
    persist_directory=persisted_directory,
)

retriver = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 10})

system_prompt = """
    Be a place to amazon guider for places to eat, visit, service and rest. Answer in the same language the following question.
    use only this knowledge to answer and don't use external sources from this context. If you don't know the answer, just say that you don't know in
    the same language of the question and try to help the user ask question about you and your knowledge. 
    If you have the answer, use the maximum amount of information you can with bullet points and list the same language of the question. If your data is another language,
    translate you data to the language of the question. All your answer must be in the same language of the question and you have to translate it.
    Context: {context}
    """

prompt = ChatPromptTemplate.from_messages([

    ('system', system_prompt),
    ('human', '{input}'),

])
question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,

)

chain = create_retrieval_chain(
    retriever=retriver,
    combine_docs_chain=question_answer_chain,

)


query = 'What is all the place that I can eat in Manacapuru?'


response = chain.invoke({'input': query},)
print(response["answer"])

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
