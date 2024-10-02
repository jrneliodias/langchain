
import os
import dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

set_llm_cache(SQLiteCache(database_path="langchain.db"))

# model = OpenAI()
model = ChatOpenAI(
    model="gpt-4",
    temperature=0.9)

db = SQLDatabase.from_uri("sqlite:///database.db")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

prompt_knowledge = PromptTemplate(
    input_variables=["question, context"],
    template="""
    Be a place to eat guider. Answer in the same language the following question
    use only this knowledge to answer and don't use external sources from this context.
    Question: {question}
    """

)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True
)

question = 'Quais restaurantes eu posso comer?'

output = agent_executor.invoke({
    'input': prompt_knowledge.format(question=question)
})

print(output.get('output'))
