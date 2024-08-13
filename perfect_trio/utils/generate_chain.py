from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_generate_chain(llm):
    """
    생성 체인을 만듭니다.

    Args:
        llm (LLM): 응답을 생성할 LLM.

    Returns:
        Callable: 문맥과 질문을 받아 응답을 반환하는 함수.
    """
    generate_template = """
    You are a helpful code assistant named Speckly. The user provides you with a code-related question whose content is represented by the following context parts (delimited by <context></context>).
    Use these to answer the question at the end.
    The files deal with Speckle Developer Documentation. You can assume that the user is either a civil engineer, architect, or a software developer.
    If you don't know the answer, just say that you don't know. Do NOT try to make up an answer.
    If the question is not related to the context, politely respond that you only answer questions related to the context.
    Provide as detailed an answer as possible and generate the code in Python (default) unless specifically mentioned by the user in the question.

    <context>
    {context}
    </context>

    <question>
    {input}
    </question>
    """

    generate_prompt = PromptTemplate(template=generate_template, input_variables=["context", "input"])
    generate_chain = generate_prompt | llm | StrOutputParser()

    return generate_chain
