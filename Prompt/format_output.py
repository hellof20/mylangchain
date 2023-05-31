# # # 格式化输出
response_schemas = [
    ResponseSchema(name="type", description="Get the instruction type from the input, the types are location, level, money"),
    ResponseSchema(name="map", description="This is your response, the map number"),
    ResponseSchema(name="x", description="This is your response, the first coordinates number"),
    ResponseSchema(name="y", description="This is your response, the second coordinates number"),
    ResponseSchema(name="z", description="This is your response, the third coordinates number"),
    ResponseSchema(name="orientation", description="This is your response, the orientation number")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

template = """
Extract the coordinates, the map and the orientation from the input coordinates respectively.
{format_instructions}
% USER INPUT:
{user_input}
YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

promptValue = prompt.format(user_input="Level Up 5")
llm_output = llm(promptValue)

# 使用解析器进行解析生成的内容
print(output_parser.parse(llm_output))