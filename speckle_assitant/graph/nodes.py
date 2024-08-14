from typing import Dict, Any
from langchain.schema import BaseRetriever
from .state import GraphState

class GraphNodes:
    def __init__(self, retriever: BaseRetriever, generate_chain, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter):
        self.retriever = retriever
        self.generate_chain = generate_chain
        self.retrieval_grader = retrieval_grader
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator
        self.question_rewriter = question_rewriter

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        print("---RETRIEVE---")
        question = state["input"]
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "input": question}

    def generate(self, state: GraphState) -> Dict[str, Any]:
        print("---GENERATE---")
        question = state["input"]
        documents = state["documents"]
        generation = self.generate_chain.invoke({"context": documents, "input": question})
        return {"documents": documents, "input": question, "generation": generation}

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["input"]
        documents = state["documents"]
        filtered_docs = [
            doc for doc in documents
            if self.retrieval_grader.grade(doc.page_content, question)["score"] == "yes"
        ]
        return {"documents": filtered_docs, "input": question}

    def transform_query(self, state: GraphState) -> Dict[str, Any]:
        print("---TRANSFORM QUERY---")
        question = state["input"]
        better_question = self.question_rewriter.rewrite(question)
        return {"documents": state["documents"], "input": better_question}