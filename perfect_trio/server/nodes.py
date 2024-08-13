import sys
import os

# 현재 파일의 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.generate_chain import create_generate_chain

class GraphNodes:
    def __init__(self, llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter):
        self.llm = llm
        self.retriever = retriever
        self.retrieval_grader = retrieval_grader
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator
        self.question_rewriter = question_rewriter
        self.generate_chain = create_generate_chain(llm)

    def retrieve(self, state):
        question = state["input"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "input": question}

    def generate(self, state):
        question = state["input"]
        documents = state["documents"]
        generation = self.generate_chain.invoke({"context": documents, "input": question})
        return {"documents": documents, "input": question, "generation": generation}

    def grade_documents(self, state):
        question = state["input"]
        documents = state["documents"]
        filtered_docs = []

        for d in documents:
            score = self.retrieval_grader.invoke({"input": question, "document": d.page_content})
            if score["score"] == "yes":
                filtered_docs.append(d)

        return {"documents": filtered_docs, "input": question}

    def transform_query(self, state):
        question = state["input"]
        better_question = self.question_rewriter.invoke({"input": question})
        return {"documents": state["documents"], "input": better_question}
