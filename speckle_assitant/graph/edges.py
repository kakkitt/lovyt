from typing import Dict, Any
from .state import GraphState

class EdgeGraph:
    def __init__(self, hallucination_grader, code_evaluator):
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator

    def decide_to_generate(self, state: GraphState) -> str:
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        print("---CHECK HALLUCINATIONS---")
        question = state["input"]
        documents = state["documents"]
        generation = state["generation"]

        hallucination_score = self.hallucination_grader.grade(generation, "\n".join([doc.page_content for doc in documents]))
        
        if hallucination_score["score"] == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            code_eval_score = self.code_evaluator.evaluate(generation, question, "\n".join([doc.page_content for doc in documents]))
            if code_eval_score["score"] == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATIONS ARE HALLUCINATED, RE-TRY---")
            return "not supported"