import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os

# CSV 파일 경로
TYPES_FILE = r"C:\Users\Lenovo\git-workspace\forGDG\2회차\과제\pokemon_types.csv"
FEEDBACK_FILE = r"C:\Users\Lenovo\git-workspace\forGDG\2회차\과제\feedback.csv"

# 한국어 불용어 리스트
korean_stopwords = set([
    "나는", "내가", "네가", "그는", "그녀는", "우리는", "너희는", "그들은",
    "이", "저", "그", "이것", "저것", "그것", "여기", "저기", "거기"
])

# SentenceTransformer 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# CSV 파일 읽기
def load_csv_files():
    # 포켓몬 타입 설명 로드
    pokemon_df = pd.read_csv(TYPES_FILE, encoding='utf-8')
    pokemon_types = pokemon_df['type'].tolist()
    type_descriptions = pokemon_df['description'].tolist()

    # 타입 설명 임베딩
    type_embeddings = model.encode(type_descriptions)

    # 피드백 파일 초기화
    if not os.path.exists(FEEDBACK_FILE):
        pd.DataFrame(columns=["user_input", "predicted_type", "corrected_type"]).to_csv(FEEDBACK_FILE, index=False)
    
    return pokemon_types, type_embeddings

# 텍스트 전처리
def preprocess_text(text):
    text = re.sub(r'[^가-힣\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in korean_stopwords]
    return ' '.join(words)

# 유사 타입 찾기
def find_similar_type(input_text, pokemon_types, type_embeddings):
    cleaned_text = preprocess_text(input_text)
    print(f"전처리된 텍스트: {cleaned_text}")
    input_embedding = model.encode([cleaned_text])
    similarities = cosine_similarity(input_embedding, type_embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:3]
    top_types = [pokemon_types[i] for i in top_indices]
    top_scores = [similarities[0][i] for i in top_indices]
    print(f"Top 3 유사 타입: {list(zip(top_types, top_scores))}")
    return top_types[0]

# 피드백 저장
def save_feedback(user_input, predicted_type, corrected_type):
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    new_entry = {"user_input": user_input, "predicted_type": predicted_type, "corrected_type": corrected_type}
    feedback_df = pd.concat([feedback_df, pd.DataFrame([new_entry])], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_FILE, index=False)
    print(f"피드백이 저장되었습니다: {new_entry}")

# 메인 실행 함수
def main():
    pokemon_types, type_embeddings = load_csv_files()
    
    while True:
        user_input = input("\n사용자 입력 (종료하려면 'esc' 입력): ").strip()

        # 'esc' 입력 시 종료
        if user_input.lower() == 'esc':
            print("프로그램을 종료합니다.")
            break
        
        predicted_type = find_similar_type(user_input, pokemon_types, type_embeddings)
        print(f"유사한 포켓몬 타입: {predicted_type}")
        
        # 피드백 입력 받기
        corrected_type = input("의도한 타입을 입력하세요 (없으면 엔터): ").strip()
        if corrected_type.lower() == 'esc':
            print("프로그램을 종료합니다.")
            break
        
        if corrected_type and corrected_type in pokemon_types:
            save_feedback(user_input, predicted_type, corrected_type)
        else:
            print("피드백이 제공되지 않았습니다.")


if __name__ == "__main__":
    main()
