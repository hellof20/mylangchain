from langchain.llms import VertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain


llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,)

# 第一条链
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
# print(chain.run("colorful socks"))

# 第二条链
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase for the following company: {company_name}",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)    

# 组合这两条链
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)