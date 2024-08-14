from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import GraphNodes
from .edges import EdgeGraph

def build_graph(nodes: GraphNodes, edges: EdgeGraph):
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)
    
    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        edges.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        edges.grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    
    # Compile
    return workflow.compile()