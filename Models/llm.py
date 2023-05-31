from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain

llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

#通过LLMChain来运行llm model
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
print(llm_chain.run(question))
