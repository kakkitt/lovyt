from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class RetrievalGrader:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model = ChatOpenAI(model_name=model_name, temperature=0)
        self.prompt = PromptTemplate(
            template="""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>

            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {input} \n
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["document", "input"],
        )
        self.chain = self.prompt | self.model | JsonOutputParser()

    def grade(self, document: str, question: str) -> dict:
        return self.chain.invoke({"document": document, "input": question})