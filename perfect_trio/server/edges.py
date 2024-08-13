class EdgeGraph:
    def __init__(self, hallucination_grader, code_evaluator):
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator

    def decide_to_generate(self, state):
        filtered_documents = state["documents"]

        if not filtered_documents:
            return "transform_query"
        else:
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        if score["score"] == "yes":
            score = self.code_evaluator.invoke({"input": state["input"], "generation": generation, "documents": documents})
            if score["score"] == "yes":
                return "useful"
            else:
                return "not useful"
        else:
            return "not supported"
