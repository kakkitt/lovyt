from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QuestionRewriter:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model = ChatOpenAI(model_name=model_name, temperature=0)
        self.prompt = PromptTemplate(
            template="""Rewrite the following question to improve its clarity and relevance:
            
            {input}
            
            Rewritten question:""",
            input_variables=["input"],
        )
        self.chain = self.prompt | self.model | StrOutputParser()

    def rewrite(self, question: str) -> str:
        return self.chain.invoke({"input": question})