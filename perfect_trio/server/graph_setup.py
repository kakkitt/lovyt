from langgraph.graph import StateGraph
from nodes import GraphNodes
from edges import EdgeGraph

def create_chain(llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter):
    workflow = StateGraph(Input)
    graph_nodes = GraphNodes(llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter)
    edge_graph = EdgeGraph(hallucination_grader, code_evaluator)

    workflow.add_node("retrieve", graph_nodes.retrieve)
    workflow.add_node("grade_documents", graph_nodes.grade_documents)
    workflow.add_node("generate", graph_nodes.generate)
    workflow.add_node("transform_query", graph_nodes.transform_query)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        edge_graph.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        edge_graph.grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()
