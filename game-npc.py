from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import TextLoader
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA
# from langchain.vectorstores import Chroma
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import StructuredTool

import requests,time
from flask import Flask, jsonify, request

app = Flask(__name__)
llm=VertexAI()

# 引用redis已经存在index
embeddings = VertexAIEmbeddings()
rds = Redis.from_existing_index(embeddings, redis_url="redis://localhost:6379", index_name='pwmlink')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_rerank", retriever=rds.as_retriever(), return_source_documents=False)

def tele(player, location):
    """teleport the player to there. Do not run when player ask about it."""
    command = "tele name %s %s" %(player['name'], location['name'])
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
    return ''
    # return 'tele name sdfasdf %f %f %f %f %f' %(map,x,y,z,orientation)

tools = [Tool(
        name = "coordinates of teleport",
        func=qa.run,
        description="useful for when need to ansower question about the teleport point information like coordinates,map and orientation. Input should be a fully formed question.",
        return_direct=True
    ),StructuredTool.from_function(tele)]
agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

@app.route("/chat", methods=["POST"])
def chat():
    player_name = request.get_json()["player_name"]
    msg = request.get_json()["msg"]
    msg = """Im a game player, my name is %s, %s""" %(player_name, msg)
    result = agent.run(msg)
    return result

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")