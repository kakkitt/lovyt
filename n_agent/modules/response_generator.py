from modules.rag_engine import RAGEngine
from modules.prompt_manager import PromptManager
from modules.content_analyzer import ContentAnalyzer

class ResponseGenerator:
    def __init__(self):
        self.rag_engine = RAGEngine()
        self.prompt_manager = PromptManager()
        self.content_analyzer = ContentAnalyzer()
    
    def generate_response(self, user_input, context=None):
        # 1. 사용자 입력 분석
        analysis = self.content_analyzer.analyze_text(user_input)
        entities = self.content_analyzer.extract_entities(user_input)
        detected_genre = self.content_analyzer.detect_genre(user_input)
        
        # 2. RAG를 사용하여 관련 정보 검색
        retrieved_info = self.rag_engine.retrieve_relevant_info(user_input)
        
        # 3. 프롬프트 생성
        if "character" in user_input.lower():
            prompt = self.prompt_manager.get_prompt("character_creation",
                                                    genre=detected_genre,
                                                    role="주요 등장인물",
                                                    traits=", ".join(entities))
        elif "plot" in user_input.lower():
            prompt = self.prompt_manager.get_prompt("plot_development",
                                                    genre=detected_genre,
                                                    main_event=user_input,
                                                    conflict="미정")
        else:
            prompt = user_input
        
        # 4. 최종 응답 생성
        response = self.rag_engine.generate_response(prompt, retrieved_info)
        
        # 5. 응답 강화