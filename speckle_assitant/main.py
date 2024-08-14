from langchain_openai import ChatOpenAI
from utils import load_api_keys, set_api_keys
from document_processing import DocumentLoader, create_vector_store
from chains import create_retrieval_chain, create_generate_chain
from graders import RetrievalGrader, HallucinationGrader, CodeEvaluator, QuestionRewriter
from graph import GraphNodes, EdgeGraph, build_graph

def main():
    # Load and set API keys
    api_keys = load_api_keys()
    set_api_keys(api_keys)

    # Initialize document loader and load documents
    doc_loader = DocumentLoader(api_keys['FIRE_API_KEY'])
    docs = doc_loader.load_saved_docs("data/crawled_docs/saved_docs.pkl")

    # Create vector store and retriever
    vector_store = create_vector_store(docs)
    retriever = vector_store.as_retriever()

    # Initialize language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Create chains
    retrieval_chain = create_retrieval_chain(retriever)
    generate_chain = create_generate_chain()

    # Initialize graders
    retrieval_grader = RetrievalGrader()
    hallucination_grader = HallucinationGrader()
    code_evaluator = CodeEvaluator()
    question_rewriter = QuestionRewriter()

    # Create graph components
    graph_nodes = GraphNodes(retriever, generate_chain, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter)
    edge_graph = EdgeGraph(hallucination_grader, code_evaluator)

    # Build the graph
    graph = build_graph(graph_nodes, edge_graph)

    return graph

if __name__ == "__main__":
    graph = main()
    print("Graph built successfully. Ready to use in server.py")