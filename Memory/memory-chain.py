from langchain.llms import VertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,)

template = """
    The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context. 
    If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
{input}
"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)
conversation = ConversationChain(
    # prompt=prompt,
    llm=llm, 
    verbose=True
)

while True:
    msg = input("input: ")
    result = conversation.run(msg)
    print(result)