from typing import List
from langchain_community.document_loaders import FireCrawlLoader
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_docs(self, url: str) -> List[Document]:
        """
        Retrieves documents from the specified URL using the FireCrawlLoader.

        Args:
            url (str): The URL to crawl for documents.

        Returns:
            List[Document]: A list of Document objects containing the retrieved content.
        """
        loader = FireCrawlLoader(
            api_key=self.api_key, url=url, mode="crawl"
        )

        raw_docs = loader.load()
        docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_docs]

        return docs

    def load_saved_docs(self, file_path: str) -> List[Document]:
        """
        Loads previously saved documents from a pickle file.

        Args:
            file_path (str): The path to the saved documents file.

        Returns:
            List[Document]: A list of Document objects containing the loaded content.
        """
        import pickle
        with open(file_path, "rb") as f:
            saved_docs = pickle.load(f)
        return saved_docs