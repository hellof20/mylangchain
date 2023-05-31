from langchain.embeddings import VertexAIEmbeddings

embeddings = VertexAIEmbeddings()
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result)
print('-------------------------------------------')
doc_result = embeddings.embed_documents([text])
print(doc_result)

# 这两个的结果是一样的