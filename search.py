from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import VertexAI
from langchain.agents import AgentType

llm = VertexAI()

 # 加载 serpapi 工具
tools = load_tools(["serpapi"])

# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=False)

# 运行 agent
result = agent.run("What's the date today? What great events have taken place today in history?")
print(result)
