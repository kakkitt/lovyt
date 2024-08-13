from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client

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
        Creates a question rewriter chain that rewrites a given question to improve its clarity and relevance.

        Returns:
            A callable function that takes a question as input and returns the rewritten question as a string.
        """
        # LangSmith Client를 사용하여 프롬프트 가져오기
        client = Client()
        re_write_prompt = client.pull_prompt("efriis/self-rag-question-rewriter")

        # 가져온 리라이트 프롬프트와 모델을 결합하여 질문 리라이터 체인 생성
        question_rewriter = re_write_prompt | self.model | StrOutputParser()

        return question_rewriter
