from typing import List
from langchain_community.document_loaders import FireCrawlLoader
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_docs(self, url: str) -> List[Document]:
        """
        지정된 URL에서 문서를 가져옵니다.

        Args:
            url (str): 문서를 크롤링할 URL.

        Returns:
            List[Document]: 가져온 문서 목록.
        """
        loader = FireCrawlLoader(
            api_key=self.api_key, url=url, mode="crawl"
        )

        raw_docs = loader.load()
        docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_docs]

        return docs
