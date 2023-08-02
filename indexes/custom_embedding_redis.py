from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import TextLoader
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

llm=VertexAI(
    model_name='text-bison@001',
    max_output_tokens=1024,
    verbose=True)
embeddings = HuggingFaceEmbeddings(model_name='distiluse-base-multilingual-cased-v1')

# 新建redis index
with open('game_tele1_cn.txt') as f:
    game_tele = f.read()
text_splitter = CharacterTextSplitter(separator = "\n",chunk_size=100, chunk_overlap=0)
docs = text_splitter.create_documents([game_tele])
print(f'documents:{len(docs)}')
rds = Redis.from_documents(docs, embeddings, redis_url="redis://localhost:6379", index_name='cn_teleport_4')

# 使用已有的redis index
# rds = Redis.from_existing_index(embeddings, redis_url="redis://localhost:6379", index_name='cn_teleport_2')

# RetrievalQA
qa = RetrievalQA.from_chain_type(llm=VertexAI(), chain_type="map_rerank", retriever=rds.as_retriever(), return_source_documents=False)
while True:
    msg = input("input: ")
    result = qa({"query": msg})
    print(result['result'])