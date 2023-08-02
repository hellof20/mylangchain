from langchain.embeddings import VertexAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain         import LLMChain
from langchain.llms    import VertexAI,OpenAI
from langchain.chains  import RetrievalQA,RetrievalQAWithSourcesChain,ConversationalRetrievalChain
from langchain.prompts import PromptTemplate,MessagesPlaceholder
from langchain.agents  import initialize_agent,AgentType,Tool,ZeroShotAgent,AgentExecutor,load_tools
from langchain.tools   import StructuredTool
from langchain.memory  import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

import requests,time,os
from flask import Flask, jsonify, request

app = Flask(__name__)

llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=1024,
    verbose=True)
# llm = OpenAI(temperature=0,max_tokens=2048)   
search = SerpAPIWrapper()

# VertexAIEmbeddings
# embeddings = VertexAIEmbeddings()

# 自定义embedding model
embeddings = HuggingFaceEmbeddings(model_name='distiluse-base-multilingual-cased-v1')

## memory
chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 引用redis已经存在index
# rds = Redis.from_existing_index(embeddings, redis_url="redis://localhost:6379", index_name='cn_teleport_4')

# 新建redis index
with open('indexes/game_tele1_cn.txt') as f:
    game_tele = f.read()
text_splitter = CharacterTextSplitter(separator = "\n",chunk_size=100, chunk_overlap=0)
docs = text_splitter.create_documents([game_tele])
print(f'documents:{len(docs)}')
rds = Redis.from_documents(docs, embeddings, redis_url="redis://localhost:6379", index_name='cn_teleport_4')


qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_rerank", retriever=rds.as_retriever(), return_source_documents=False)


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
    url="http://pwm:123@localhost:7878/"
    headers = {'content-type': 'text/xml'}
    response = requests.post(url,data=body,headers=headers)

def teleport(player_name:str, location_name:str):
    # """when player asked to or want to teleport, then teleport player to the location."""
    command = "tele name %s %s" %(player_name, location_name.replace(' ',''))
    # print(command)
    soap_request(command)
    return ''

def player_level(player_name:str, level_number:int):
    """player level number."""
    command = "character level %s %d" %(player_name, level_number)
    print(command)
    soap_request(command)
    return 'change level success!'

def parsing_level(string):
    a, b = string.split(",")
    player_level(a,int(b))
    return 'change level success'

def parsing_teleport(string):
    # print("--------------------------------------")
    # print(string)
    # print("--------------------------------------")
    a, b = string.split(",")
    teleport(a,b)
    return 'teleport success'

parsing_teleport = Tool(
    name = "teleport player",
    func = parsing_teleport,
    # description = "need player name and English name of the teleport, when player want to teleport or go there or go back"
    description = "当玩家想去哪里或者传送或者回到某个地方的时候，提取玩家名称和坐标位置的英文名称"
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
    # description = "当提问的内容中含有坐标或者询问什么地方在哪里。输入是完整的问题",
    return_direct=False
)

# https://python.langchain.com/en/latest/modules/agents/agents/examples/conversational_agent.html
current_search = Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world or current date or current weather"
    )
tools = [location_search, parsing_teleport, parsing_level, current_search]

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
    return response



background = """
你是玩家在游戏《魔兽世界》中的助手，你可以帮助玩家完成以下任务:
- 解答关于游戏的各种问题
- 完成位置传送
- 改变玩家等级
- 生成装备等任务
"""
agent.run(background)

while True:
   msg = input("input: ")
   result = agent.run(msg)
   print(result)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")