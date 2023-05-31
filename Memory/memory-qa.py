# https://python.langchain.com/en/latest/modules/memory/examples/adding_memory_chain_multiple_inputs.html
from langchain.embeddings import VertexAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.docstore.document import Document
from langchain.vectorstores.redis import Redis
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

llm=VertexAI()

# 引用redis已经存在index
embeddings = VertexAIEmbeddings()
rds = Redis.from_existing_index(embeddings, redis_url="redis://localhost:6379", index_name='pwmlink')
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_rerank", retriever=rds.as_retriever(), return_source_documents=False)


template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_chain(VertexAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)

query = "What did the president say about Justice Breyer"
chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)

print(chain.memory.buffer)