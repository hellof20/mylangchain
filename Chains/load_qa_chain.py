from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import VertexAI
from langchain.vectorstores.redis import Redis

embeddings = VertexAIEmbeddings()
llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,)


rds = Redis.from_existing_index(embeddings, redis_url="redis://localhost:6379", index_name='pwmlink')
query = "do you know the coordinates of teleport HelmsBedLake?"

# 直接通过redis搜索问题答案
docs = rds.similarity_search(query)
print(docs[0].page_content)

# 通过qa_chain来redis搜索到的结果发给大语言模型处理
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
print(qa_chain.run(input_documents=docs, question=query))


