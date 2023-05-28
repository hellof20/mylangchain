from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import TextLoader
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA


llm=VertexAI()
# loader = TextLoader('game_tele1.txt')
# documents = loader.load()
# docs = text_splitter.split_documents(game_tele)


# 新建redis index
with open('game_tele1.txt') as f:
    game_tele = f.read()
text_splitter = CharacterTextSplitter(separator = "\n",chunk_size=100, chunk_overlap=0)
docs = text_splitter.create_documents([game_tele])
print(f'documents:{len(docs)}')

embeddings = VertexAIEmbeddings()
x=0
while x<len(docs):
    y = x + 4
    rds = Redis.from_documents(docs[x:y], embeddings, redis_url="redis://localhost:6379",  index_name='pwmlink')
    x += 5
qa = RetrievalQA.from_chain_type(llm=VertexAI(), chain_type="map_rerank", retriever=rds.as_retriever(), return_source_documents=False)
result = qa({"query": "do you know the coordinates of Auchindoun?"})
print(result)