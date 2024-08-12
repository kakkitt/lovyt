import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

class ContentAnalyzer:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_text(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        word_freq = Counter(words)
        
        analysis = {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "unique_words": len(set(words)),
            "top_words": word_freq.most_common(10),
            "average_sentence_length": len(words) / len(sentences) if sentences else 0
        }
        
        return analysis
    
    def extract_entities(self, text):
        # 이 부분은 더 복잡한 NER(Named Entity Recognition) 모델을 사용하여 확장할 수 있습니다.
        # 여기서는 간단한 예시만 제공합니다.
        words = word_tokenize(text)
        entities = [word for word in words if word[0].isupper()]
        return list(set(entities))
    
    def detect_genre(self, text):
        # 장르 감지 로직을 구현합니다. 
        # 이 부분은 사전 정의된 키워드나 기계 학습 모델을 사용하여 더 정교하게 만들 수 있습니다.
        genre_keywords = {
            "판타지": ["마법", "드래곤", "기사", "왕국", "마법사"],
            "SF": ["우주", "로봇", "미래", "기술", "외계인"],
            "로맨스": ["사랑", "연애", "키스", "데이트", "결혼"],
            "추리": ["범죄", "탐정", "수사", "미스터리", "증거"]
        }
        
        word_freq = Counter(word_tokenize(text.lower()))
        genre_scores = {genre: sum(word_freq[keyword] for keyword in keywords if keyword in word_freq)
                        for genre, keywords in genre_keywords.items()}
        
        return max(genre_scores, key=genre_scores.get)

# 사용 예:
# analyzer = ContentAnalyzer()
# text = "용감한 기사가 드래곤을 물리치고 공주를 구했다. 마법사의 도움으로 왕국은 평화를 되찾았다."
# analysis = analyzer.analyze_text(text)
# entities = analyzer.extract_entities(text)
# genre = analyzer.detect_genre(text)
# print(analysis)
# print(f"Entities: {entities}")
# print(f"Detected genre: {genre}")