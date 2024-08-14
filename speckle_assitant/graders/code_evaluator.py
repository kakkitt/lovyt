from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class CodeEvaluator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model = ChatOpenAI(model_name=model_name, temperature=0)
        self.prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a code evaluator assessing whether the generated code is correct and relevant to the given question.
            Provide a JSON response with the following keys:

            'score': A binary score 'yes' or 'no' indicating whether the code is correct and relevant.
            'feedback': A brief explanation of your evaluation, including any issues or improvements needed.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the generated code:
            \n ------- \n
            {generation}
            \n ------- \n
            Here is the question: {input}
            \n ------- \n
            Here are the relevant documents: {documents}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "input", "documents"],
        )
        self.chain = self.prompt | self.model | JsonOutputParser()

    def evaluate(self, generation: str, input: str, documents: str) -> dict:
        return self.chain.invoke({"generation": generation, "input": input, "documents": documents})