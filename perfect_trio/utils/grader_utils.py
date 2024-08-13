from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class GraderUtils:
    def __init__(self, model):
        self.model = model

    def create_retrieval_grader(self):
        """
        문서가 사용자 질문과 관련이 있는지 평가하는 그레이더를 생성합니다.
        """
        grade_prompt = PromptTemplate(
            template="""
            system
            You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            
            user

            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {input} \n
            
            assistant
            """,
            input_variables=["document", "input"],
        )

        retriever_grader = grade_prompt | self.model | JsonOutputParser()
        return retriever_grader

    def create_hallucination_grader(self):
        """
        생성된 답변이 문서의 사실과 일치하는지 평가하는 그레이더를 생성합니다.
        """
        hallucination_prompt = PromptTemplate(
            template="""system
            You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            
            user
            Here are the facts:
            \n ------- \n
            {documents}
            \n ------- \n
            Here is the answer: {generation}
            
            assistant""",
            input_variables=["generation", "documents"],
        )

        hallucination_grader = hallucination_prompt | self.model | JsonOutputParser()
        return hallucination_grader

    def create_code_evaluator(self):
        """
        생성된 코드가 올바르고 주어진 질문에 관련이 있는지 평가하는 그레이더를 생성합니다.
        """
        eval_template = PromptTemplate(
            template="""system You are a code evaluator assessing whether the generated code is correct and relevant to the given question.
            Provide a JSON response with the following keys:

            'score': A binary score 'yes' or 'no' indicating whether the code is correct and relevant.
            'feedback': A brief explanation of your evaluation, including any issues or improvements needed.

            user
            Here is the generated code:
            \n ------- \n
            {generation}
            \n ------- \n
            Here is the question: {input}
            \n ------- \n
            Here are the relevant documents: {documents}
            assistant""",
            input_variables=["generation", "input", "documents"],
        )

        code_evaluator = eval_template | self.model | JsonOutputParser()
        return code_evaluator

    def create_question_rewriter(self):
        """
        질문을 더 명확하고 관련성 있게 다시 작성하는 리라이터 체인을 생성합니다.
        """
        re_write_prompt = hub.pull("efriis/self-rag-question-rewriter")
        question_rewriter = re_write_prompt | self.model | StrOutputParser()

        return question_rewriter
