from langchain.chat_models import ChatVertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain import LLMChain

chat = ChatVertexAI()

# messages = [
#     SystemMessage(content="You are a helpful assistant that translates English to French."),
#     HumanMessage(content="Translate this sentence from English to French. I love programming.")
# ]
# print(chat(messages))

#
template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 用chat模型直接运行
message = chat_prompt.format_prompt(input_language="English", output_language="French", text="Translate this sentence from English to French. I love programming.").to_messages()
print(chat(message))
print('-------------------------------------------')


# 用chain来运行
chain = LLMChain(llm=chat, prompt=chat_prompt)
print(chain.run({
    'input_language': "English", 
    'output_language': "French", 
    'text': "Translate this sentence from English to French. I love programming."
}))