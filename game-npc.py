from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import TextLoader
from langchain.llms import VertexAI
from langchain import LLMChain
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain,ConversationalRetrievalChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent,AgentType,Tool,ZeroShotAgent,AgentExecutor
from langchain.tools import StructuredTool
from langchain.memory import ChatMessageHistory,ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

import requests,time
from flask import Flask, jsonify, request

app = Flask(__name__)
llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=1024,
    temperature=0,
    top_p=0.8,
    top_k=40,
    verbose=True)

embeddings = VertexAIEmbeddings()

#https://python.langchain.com/en/latest/modules/agents/agents/examples/structured_chat.html#adding-in-memory
## memory
chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 引用redis已经存在index
rds = Redis.from_existing_index(embeddings, redis_url="redis://localhost:6379", index_name='pwmlink')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=rds.as_retriever(), return_source_documents=False)

def soap_request(command):
    body = """<SOAP-ENV:Envelope
    xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" 
    xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" 
    xmlns:xsi="http://www.w3.org/1999/XMLSchema-instance" 
    xmlns:xsd="http://www.w3.org/1999/XMLSchema" 
    xmlns:ns1="urn:AC">
    <SOAP-ENV:Body>
	<ns1:executeCommand>
	    <command>%s</command>
	</ns1:executeCommand>
    </SOAP-ENV:Body>
    </SOAP-ENV:Envelope>""" % command
    url="http://pwm:pwm123@localhost:7878/"
    headers = {'content-type': 'text/xml'}
    response = requests.post(url,data=body,headers=headers)


def teleport(player_name:str, location_name:str):
    """when player asked to or want to teleport, then teleport player to the location."""
    command = "tele name %s %s" %(player_name, location_name.replace(' ',''))
    print(command)
    # soap_request(command)
    return ''

def player_level(player_name:str, level_number:int):
    """player level number."""
    command = "character level %s %d" %(player_name, level_number)
    print(command)
    # soap_request(command)
    return 'change level success!'

def parsing_level(string):
    print(string)
    a, b = string.split(",")
    player_level(a,int(b))
    return 'change level success'

def parsing_teleport(string):
    print(string)
    a, b = string.split(",")
    teleport(a,b)
    return 'teleport success'

parsing_teleport = Tool(
    name = "teleport player",
    func = parsing_teleport,
    description = "need player name and location, when player want to teleport or go there"
)

parsing_level = Tool(
    name = "player level",
    func = parsing_level,
    description = "need player name and level number, useful when palyer asked to or want to modfy their level"
)

# teleport = StructuredTool.from_function(teleport)
# player_level = StructuredTool.from_function(player_level)
location_search = Tool(
    name = "location search",
    func=qa.run,
    description = "when player ask to give teleport or tele, questions about map, coordinates, orientation of teleport or tele. input is the complete questions and ask.",
    return_direct=False
)

# tools = [location_search, teleport, player_level]
tools = [location_search, parsing_teleport, parsing_level]

# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     handle_parsing_errors=True,
#     memory=memory, 
#     agent_kwargs = {
#         "memory_prompts": [chat_history],
#         "input_variables": ["input", "agent_scratchpad", "chat_history"]
#     },
#     verbose=True)
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)


## api
@app.route("/chat", methods=["POST"])
def chat():
    player_name = request.get_json()["player_name"]
    msg = request.get_json()["msg"]
    response= agent.run(msg)
    # try:
    #     response= agent.run(msg)
    # except Exception as e:
    #     response = str(e)
    #     if response.startswith("Could not parse LLM output"):
    #         response = 'try again'
    return response

while True:
    msg = input("input: ")
    result = agent.run(msg)
    print(result)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")