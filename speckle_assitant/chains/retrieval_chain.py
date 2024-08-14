from langchain.schema import BaseRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def create_retrieval_chain(retriever: BaseRetriever, model_name: str = "gpt-3.5-turbo") -> RetrievalQA:
    """
    Creates a retrieval chain for question answering.

    Args:
        retriever (BaseRetriever): The retriever to use for document lookup.
        model_name (str): The name of the language model to use.

    Returns:
        RetrievalQA: A retrieval QA chain.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return chain