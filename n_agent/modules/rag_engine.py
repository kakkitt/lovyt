import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGEngine:
    def __init__(self, model_name="klue/bert-base", knowledge_base_path="data/knowledge_base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        
    def load_knowledge_base(self, path):
        knowledge_base = {}
        for file in ["genre_info.json", "character_archetypes.json", "plot_structures.json"]:
            with open(f"{path}/{file}", "r", encoding="utf-8") as f:
                knowledge_base.update(json.load(f))
        return knowledge_base
    
    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def retrieve_relevant_info(self, query, top_k=3):
        query_embedding = self.encode_text(query)
        similarities = []
        
        for key, value in self.knowledge_base.items():
            value_embedding = self.encode_text(value)
            similarity = cosine_similarity(query_embedding, value_embedding)[0][0]
            similarities.append((key, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(key, self.knowledge_base[key]) for key, _ in similarities[:top_k]]
                                                                    
    def generate_response(self, query, retrieved_info):
        # 여기에 검색된 정보를 바탕으로 응답을 생성하는 로직을 구현합니다.
        # 이 부분은 프롬프트 엔지니어링과 결합하여 더 정교하게 만들 수 있습니다.
        response = f"Query: {query}\n\nRelevant information:\n"
        for key, value in retrieved_info:
            response += f"- {key}: {value}\n"
        return response

# 사용 예:
# rag_engine = RAGEngine()
# query = "판타지 장르의 주인공 캐릭터 설정에 대해 조언해주세요."
# retrieved_info = rag_engine.retrieve_relevant_info(query)
# response = rag_engine.generate_response(query, retrieved_info)
# print(response)