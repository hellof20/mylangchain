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
llm=VertexAI(temperature=0)

## memory
chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 引用redis已经存在index
embeddings = VertexAIEmbeddings()
rds = Redis.from_existing_index(embeddings, redis_url="redis://localhost:6379", index_name='pwmlink')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_rerank", 
retriever=rds.as_retriever(), return_source_documents=False)

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

def teleport(player_name, location_name):
    """teleport player to the location."""
    command = "tele name %s %s" %(player_name, location_name)
    print(command)
    soap_request(command)
    return ''

def change_player_level(player_name, level_number:float):
    """change player to level number."""
    command = "character level %s %d" %(player_name, level_number)
    print(command)
    soap_request(command)
    return ''

tools = [Tool(
        name = "location search",
        func=qa.run,
        # description="useful for when you need to answer questions about teleport map or teleport coordinates",
        description = "only for teleport questions. input is the complete questions and ask.",
        return_direct=False
        ),StructuredTool.from_function(teleport),StructuredTool.from_function(change_player_level)]

# tools = [StructuredTool.from_function(teleport)]

agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
     verbose=True,
     memory=memory, 
     agent_kwargs = {
        "memory_prompts": [chat_history],
        "input_variables": ["input", "agent_scratchpad", "chat_history"]
    })
#agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)


## api
@app.route("/chat", methods=["POST"])
def chat():
    player_name = request.get_json()["player_name"]
    msg = request.get_json()["msg"]
    result = agent.run(msg)
    return result

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")