import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from autointmlp import AutoIntMLPModel

# ===============================
# 기본 경로 설정
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ML1M_DIR = os.path.join(DATA_DIR, "ml-1m")
MODEL_DIR = os.path.join(BASE_DIR, "model")

def safe_encode(encoder, value, default=None):
    """
    LabelEncoder에 없는 값이 들어오면 default 값으로 대체
    """
    value = str(value)

    if value in encoder.classes_:
        return encoder.transform([value])[0]

    if default is None:
        default = encoder.classes_[0]

    return encoder.transform([default])[0]


# ===============================
# 모델 + 데이터 로드
# ===============================
@st.cache_resource
def load_model_and_data():
    tf.keras.backend.clear_session()

    # field_dims 로드
    field_dims = np.load(os.path.join(DATA_DIR, "field_dims_mlp.npy"))

    # LabelEncoder 로드
    label_encoders = joblib.load(
        os.path.join(DATA_DIR, "autoIntMLP_label_encoders.pkl")
    )

    # 모델 생성
    embedding_size=32
    model = AutoIntMLPModel(
        field_dims=field_dims,
        embedding_size=embedding_size,
        att_layer_num=3,
        att_head_num=2,
        att_res=True,
        dnn_dropout=0.4,
        init_std=0.0001
    )

    # 더미 입력으로 모델 초기화
    dummy_input = np.zeros((1, len(field_dims)), dtype=np.int32)
    model.predict(dummy_input, verbose=0)

    # 가중치 로드
    model.load_weights(
        os.path.join(MODEL_DIR, "autoIntMLP_model_weights.weights.h5")
    )

    # ⚠️ 반드시 prepro 데이터 사용
    movies = pd.read_csv(os.path.join(ML1M_DIR, "movies_prepro.csv"))
    users = pd.read_csv(os.path.join(ML1M_DIR, "users_prepro.csv"))
    ratings = pd.read_csv(os.path.join(ML1M_DIR, "ratings_prepro.csv"))

    field_dims = np.load(os.path.join(DATA_DIR, "field_dims_mlp.npy"))
    label_encoders = joblib.load(os.path.join(DATA_DIR, "autoIntMLP_label_encoders.pkl"))
    
    return model, label_encoders, movies, users, ratings


model, label_encoders, movies_raw, users_raw, ratings_raw = load_model_and_data()

# ===============================
# UI
# ===============================
st.title("AutoIntMLP 영화 추천 시스템")

user_id_input = st.number_input(
    "👤 사용자 ID 입력 (1 ~ 6040)",
    min_value=1,
    max_value=6040,
    value=1
)

num_recommendations = st.slider(
    "🎯 추천 받을 영화 개수",
    min_value=5,
    max_value=20,
    value=10
)

# ===============================
# 추천 버튼
# ===============================
if st.button("🎥 영화 추천 받기"):
    user_id = int(user_id_input)

    # 사용자 존재 여부 확인
    if user_id not in users_raw["user_id"].values:
        st.error("❌ 존재하지 않는 사용자 ID입니다.")
        st.stop()

    st.success(f"👤 사용자 {user_id} 님을 위한 추천 결과")

    # 사용자가 이미 본 영화
    seen_movies = ratings_raw[
        ratings_raw["user_id"] == user_id
    ]["movie_id"].values

    # 추천 후보 영화 (아직 안 본 영화)
    candidate_movies = movies_raw[
        ~movies_raw["movie_id"].isin(seen_movies)
    ].copy()

    # 사용자 정보
    user = users_raw[users_raw["user_id"] == user_id].iloc[0]

    # 사용자 feature 인코딩 (safe)
    user_id_enc = safe_encode(label_encoders["user_id"], user["user_id"])
    gender_enc = safe_encode(label_encoders["gender"], user["gender"])
    age_enc = safe_encode(label_encoders["age"], user["age"])
    occupation_enc = safe_encode(label_encoders["occupation"], user["occupation"])
    zip_enc = safe_encode(label_encoders["zip"], user["zip"])

    inputs = []

    for _, row in candidate_movies.iterrows():
        features = [
            user_id_enc,
            safe_encode(label_encoders["movie_id"], row["movie_id"]),
            safe_encode(label_encoders["movie_decade"], row["movie_decade"]),
            safe_encode(label_encoders["movie_year"], row["movie_year"]),
            safe_encode(label_encoders["rating_year"], "2000"),
            safe_encode(label_encoders["rating_month"], "12"),
            safe_encode(label_encoders["rating_decade"], "2000s"),
            safe_encode(label_encoders["genre1"], row["genre1"], default="no"),
            safe_encode(label_encoders["genre2"], row["genre2"], default="no"),
            safe_encode(label_encoders["genre3"], row["genre3"], default="no"),
            gender_enc,
            age_enc,
            occupation_enc,
            zip_enc,
        ]
        inputs.append(features)

    # 모델 입력 배열
    inputs = np.array(inputs, dtype=np.int32)

    # 예측
    scores = model.predict(inputs, verbose=0).flatten()

    # 결과 정리
    result = candidate_movies.copy()
    result["예측 선호도 점수"] = scores

    top_n = result.sort_values(
        by="예측 선호도 점수",
        ascending=False
    ).head(num_recommendations)

    st.table(
        top_n[["title", "예측 선호도 점수"]]
        .rename(columns={"title": "🎬 영화 제목"})
    )
