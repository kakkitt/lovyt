class PromptManager:
    def __init__(self):
        self.templates = {
            "character_creation": "다음 정보를 바탕으로 흥미로운 캐릭터를 만들어주세요:\n장르: {genre}\n역할: {role}\n특징: {traits}",
            "plot_development": "다음 요소를 포함하는 흥미진진한 플롯을 제안해주세요:\n장르: {genre}\n주요 사건: {main_event}\n갈등: {conflict}",
            "world_building": "다음 요소를 고려하여 독특한 세계관을 만들어주세요:\n장르: {genre}\n핵심 요소: {key_elements}\n분위기: {atmosphere}",
            "dialogue_writing": "다음 상황에 맞는 대화를 작성해주세요:\n등장인물: {characters}\n상황: {situation}\n감정: {emotions}",
            "scene_description": "다음 정보를 바탕으로 생생한 장면을 묘사해주세요:\n장소: {location}\n시간: {time}\n분위기: {mood}"
        }
    
    def get_prompt(self, prompt_type, **kwargs):
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return self.templates[prompt_type].format(**kwargs)

    def add_template(self, name, template):
        self.templates[name] = template

    def remove_template(self, name):
        if name in self.templates:
            del self.templates[name]

# 사용 예:
# prompt_manager = PromptManager()
# character_prompt = prompt_manager.get_prompt("character_creation", 
#                                              genre="판타지", 
#                                              role="주인공", 
#                                              traits="용감하고 지혜로움")
# print(character_prompt)