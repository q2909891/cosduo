"""
스킨케어 추천 시스템 Phase 2 — v2 (최적 모델)
BERT 임베딩 + BPR + LightGCN + 동적 가중치
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import glob
import io
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchvision.transforms as T
from PIL import Image

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="스킨케어 추천 시스템 v2",
    page_icon="🧴",
    layout="wide",
)

st.markdown("""
<style>
.rec-card {
    background:#fafafa;
    border-radius:12px;
    padding:16px 20px;
    margin-bottom:12px;
    border-left:5px solid #FF6B6B;
    box-shadow:0 1px 4px rgba(0,0,0,.08);
}
.func-tag {
    display:inline-block;
    background:#eef6fb;
    border-radius:10px;
    padding:2px 8px;
    margin:2px;
    font-size:12px;
    color:#2c7da0;
}
.score-label { font-size:12px; color:#888; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SEVERITY_COLS = [
    "Acne_Severity", "Dryness_Severity", "Pigmentation_Severity",
    "Aging_Severity", "Sensitivity_Severity",
]
SEVERITY_LABELS = {
    "Acne_Severity":         "여드름/트러블",
    "Dryness_Severity":      "건조함",
    "Pigmentation_Severity": "색소침착/기미",
    "Aging_Severity":        "노화/주름",
    "Sensitivity_Severity":  "민감성",
}
SEVERITY_EMOJI = {
    "Acne_Severity":         "🔴",
    "Dryness_Severity":      "💧",
    "Pigmentation_Severity": "⭐",
    "Aging_Severity":        "⏳",
    "Sensitivity_Severity":  "🌿",
}

SEVERITY_FUNCTION_MAP = {
    "Acne_Severity":         ["SkinBarrier", "Exfoliation", "Moisturizing"],
    "Dryness_Severity":      ["Moisturizing", "SkinBarrier"],
    "Pigmentation_Severity": ["Whitning", "Exfoliation"],
    "Aging_Severity":        ["Firming", "Moisturizing", "SkinBarrier"],
    "Sensitivity_Severity":  ["SkinBarrier", "Moisturizing"],
}
FUNCTION_KOR = {
    "Moisturizing": "수분공급",
    "SkinBarrier":  "피부장벽",
    "Firming":      "탄력/리프팅",
    "Whitning":     "미백/톤업",
    "Exfoliation":  "각질케어",
}

# KNN 피처: Skin_Type_enc, Hormonal_enc 추가
KNN_FEATURE_COLS = [
    "Acne_Severity", "Dryness_Severity", "Aging_Severity",
    "Pigmentation_Severity", "Sensitivity_Severity",
    "Age", "gender", "Climate_enc", "Diet_enc",
    "Skin_Type_enc", "Hormonal_enc",
]

CATEGORIES = ["마스크팩", "스킨케어", "앰플_세럼", "크림_로션", "선케어"]
CAT_DISPLAY = {
    "마스크팩":  "마스크팩 🎭",
    "스킨케어":  "스킨케어 💆",
    "앰플_세럼": "앰플/세럼 💎",
    "크림_로션": "크림/로션 🥛",
    "선케어":    "선케어 ☀️",
}
CAT_COLOR = {
    "마스크팩":  "#FF6B6B",
    "스킨케어":  "#4ECDC4",
    "앰플_세럼": "#45B7D1",
    "크림_로션": "#96CEB4",
    "선케어":    "#FFA94D",
}

MODEL_LABELS = {
    "content":  "기능매칭 (BERT 강화)",
    "knn":      "KNN Only",
    "hybrid":   "하이브리드 KNN (권장)",
    "svd":      "SVD (넷플릭스 방식)",
    "ensemble": "앙상블 (KNN+BPR+LightGCN)",
}

MODEL_COLORS = [
    "#4ECDC4", "#45B7D1", "#FF6B6B", "#96CEB4", "#FFA94D",
    "#C77DFF", "#7B2FBE", "#06D6A0", "#118AB2", "#F72585", "#3A0CA3",
]

# ─────────────────────────────────────────────
# Baumann 16 type data
# ─────────────────────────────────────────────
BAUMANN_TYPE_DATA = {
    "OSPT": {
        "full_name": "지성·민감·색소·탄력형",
        "description": "피지 분비가 많고 민감하며, 색소침착 경향이 있지만 탄력은 유지된 타입.",
        "indicators": {"지성(O)": True, "민감성(S)": True, "색소침착(P)": True, "노화(W)": False},
        "main_concern": "여드름·트러블, 색소침착, 자극 반응",
        "care_direction": "논코메도제닉 제품 사용, 나이아신아마이드로 색소 케어, 자외선 차단 철저",
    },
    "OSPW": {
        "full_name": "지성·민감·색소·노화형",
        "description": "피지가 많고 민감하며, 색소침착과 노화 두 가지 고민을 동시에 가진 복합 타입.",
        "indicators": {"지성(O)": True, "민감성(S)": True, "색소침착(P)": True, "노화(W)": True},
        "main_concern": "여드름·트러블, 색소침착, 노화·주름, 자극 반응",
        "care_direction": "자외선 차단 최우선, 레티놀·나이아신아마이드 병용, 항염 성분으로 민감성 관리",
    },
    "OSNT": {
        "full_name": "지성·민감·무색소·탄력형",
        "description": "피지 분비가 많고 민감하지만, 색소침착과 노화 고민이 적은 타입.",
        "indicators": {"지성(O)": True, "민감성(S)": True, "색소침착(P)": False, "노화(W)": False},
        "main_concern": "여드름·트러블, 피지 조절, 자극 반응",
        "care_direction": "저자극 세정제, 유·수분 밸런스 케어, 살리실산으로 피지 조절",
    },
    "OSNW": {
        "full_name": "지성·민감·무색소·노화형",
        "description": "피지가 많고 민감하며, 색소침착은 적지만 노화 고민이 있는 타입.",
        "indicators": {"지성(O)": True, "민감성(S)": True, "색소침착(P)": False, "노화(W)": True},
        "main_concern": "여드름·트러블, 노화·주름, 자극 반응",
        "care_direction": "저자극 안티에이징 케어, 저농도 레티놀로 시작, 자외선 차단 필수",
    },
    "ORPT": {
        "full_name": "지성·저항·색소·탄력형",
        "description": "피지 분비가 많고 외부 자극에 강하며, 색소침착 경향이 있지만 탄력은 좋은 타입.",
        "indicators": {"지성(O)": True, "민감성(S)": False, "색소침착(P)": True, "노화(W)": False},
        "main_concern": "색소침착, 피지 과다, 모공 관리",
        "care_direction": "나이아신아마이드·비타민C로 미백 케어, 각질 관리로 모공 최소화",
    },
    "ORPW": {
        "full_name": "지성·저항·색소·노화형",
        "description": "피지가 많고 자극에 강하며, 색소침착과 노화가 동시에 진행되는 타입.",
        "indicators": {"지성(O)": True, "민감성(S)": False, "색소침착(P)": True, "노화(W)": True},
        "main_concern": "색소침착, 노화·주름, 피지 과다",
        "care_direction": "레티놀+비타민C 병용, SPF50+ 자외선 차단, 각질 관리",
    },
    "ORNT": {
        "full_name": "지성·저항·무색소·탄력형",
        "description": "피지가 많고 자극에 강하며, 색소침착과 노화 고민이 거의 없는 건강한 타입.",
        "indicators": {"지성(O)": True, "민감성(S)": False, "색소침착(P)": False, "노화(W)": False},
        "main_concern": "피지 과다, 모공, 여드름",
        "care_direction": "가벼운 보습 위주, 클렌징으로 피지·모공 케어, 자외선 차단으로 유지",
    },
    "ORNW": {
        "full_name": "지성·저항·무색소·노화형",
        "description": "피지가 많고 자극에 강하며, 색소침착은 적지만 노화가 진행되는 타입.",
        "indicators": {"지성(O)": True, "민감성(S)": False, "색소침착(P)": False, "노화(W)": True},
        "main_concern": "노화·주름, 피지 과다",
        "care_direction": "안티에이징 세럼 집중 케어, 레티놀 적극 활용, 자외선 차단",
    },
    "DSPT": {
        "full_name": "건성·민감·색소·탄력형",
        "description": "건조하고 민감하며, 색소침착 경향이 있지만 탄력은 유지된 타입.",
        "indicators": {"지성(O)": False, "민감성(S)": True, "색소침착(P)": True, "노화(W)": False},
        "main_concern": "건조함, 민감 반응, 색소침착",
        "care_direction": "고보습 저자극 제품, 세라마이드로 장벽 강화, 자외선 차단으로 색소 예방",
    },
    "DSPW": {
        "full_name": "건성·민감·색소·노화형",
        "description": "건조하고 민감하며, 색소침착과 노화 고민을 동시에 가진 가장 복합적인 타입.",
        "indicators": {"지성(O)": False, "민감성(S)": True, "색소침착(P)": True, "노화(W)": True},
        "main_concern": "건조함, 민감 반응, 색소침착, 노화·주름",
        "care_direction": "피부 장벽 강화 최우선, SPF50+ 자외선 차단, 순한 성분의 안티에이징 케어",
    },
    "DSNT": {
        "full_name": "건성·민감·무색소·탄력형",
        "description": "건조하고 민감하지만, 색소침착과 노화 고민이 적은 타입.",
        "indicators": {"지성(O)": False, "민감성(S)": True, "색소침착(P)": False, "노화(W)": False},
        "main_concern": "건조함, 민감 반응, 피부 장벽 약화",
        "care_direction": "세라마이드·히알루론산 집중 보습, 향료·알코올 무첨가 제품 선택",
    },
    "DSNW": {
        "full_name": "건성·민감·무색소·노화형",
        "description": "건조하고 민감하며, 색소침착은 적지만 노화가 진행되는 타입.",
        "indicators": {"지성(O)": False, "민감성(S)": True, "색소침착(P)": False, "노화(W)": True},
        "main_concern": "건조함, 노화·주름, 민감 반응",
        "care_direction": "저자극 안티에이징+고보습 복합 케어, 펩타이드 세럼 활용",
    },
    "DRPT": {
        "full_name": "건성·저항·색소·탄력형",
        "description": "건조하고 자극에 강하며, 색소침착 경향이 있지만 탄력은 좋은 타입.",
        "indicators": {"지성(O)": False, "민감성(S)": False, "색소침착(P)": True, "노화(W)": False},
        "main_concern": "건조함, 색소침착, 기미",
        "care_direction": "충분한 보습 후 나이아신아마이드·비타민C로 미백 케어, 자외선 차단",
    },
    "DRPW": {
        "full_name": "건성·저항·색소·노화형",
        "description": "건조하고 자극에 강하며, 색소침착과 노화가 동시에 진행되는 타입.",
        "indicators": {"지성(O)": False, "민감성(S)": False, "색소침착(P)": True, "노화(W)": True},
        "main_concern": "건조함, 색소침착, 노화·주름",
        "care_direction": "레티놀+나이아신아마이드 병용, 고보습 오일 추가, SPF50+ 자외선 차단",
    },
    "DRNT": {
        "full_name": "건성·저항·무색소·탄력형",
        "description": "건조하고 자극에 강하며, 색소침착과 노화 고민이 거의 없는 타입.",
        "indicators": {"지성(O)": False, "민감성(S)": False, "색소침착(P)": False, "노화(W)": False},
        "main_concern": "건조함, 수분 부족",
        "care_direction": "집중 보습 루틴, 히알루론산·글리세린 중심 케어, 자외선 차단으로 유지",
    },
    "DRNW": {
        "full_name": "건성·저항·무색소·노화형",
        "description": "건조하고 자극에 강하며, 색소침착은 적지만 노화가 진행되는 타입.",
        "indicators": {"지성(O)": False, "민감성(S)": False, "색소침착(P)": False, "노화(W)": True},
        "main_concern": "건조함, 노화·주름",
        "care_direction": "레티놀·펩타이드 세럼으로 안티에이징, 리치한 보습 크림, 자외선 차단",
    },
}

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    excel_files = glob.glob("*.xlsx")
    if not excel_files:
        st.error("*.xlsx 파일을 찾을 수 없습니다.")
        st.stop()
    excel_path = sorted(excel_files)[0]

    users = pd.read_csv("users.csv", low_memory=False)
    users[SEVERITY_COLS] = users[SEVERITY_COLS].fillna(0)

    # gender 결측값 Random Forest로 채우기
    CAT_COLS_FOR_GENDER = ["Skin_Type", "Climate", "Diet", "Hormonal_Status", "Budget_Level"]
    GENDER_FEATURES = CAT_COLS_FOR_GENDER + [
        "Age", "Acne_Severity", "Dryness_Severity", "Aging_Severity",
        "Pigmentation_Severity", "Sensitivity_Severity"
    ]
    _le = {}
    users_enc = users.copy()
    for col in CAT_COLS_FOR_GENDER:
        _le[col] = LabelEncoder()
        users_enc[col + "_enc"] = _le[col].fit_transform(users_enc[col].astype(str))
    feat_cols = [c + "_enc" if c in CAT_COLS_FOR_GENDER else c for c in GENDER_FEATURES]
    has_g = users_enc[users_enc["gender"].notna()]
    no_g  = users_enc[users_enc["gender"].isna()]
    if len(no_g) > 0:
        rf_gender = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_gender.fit(has_g[feat_cols], has_g["gender"])
        users.loc[users["gender"].isna(), "gender"] = rf_gender.predict(no_g[feat_cols])
    users["gender"] = users["gender"].fillna(0).astype(float)

    # Climate, Diet, Skin_Type, Hormonal_Status 인코딩 (KNN 피처용)
    for col in ["Climate", "Diet"]:
        le_knn = LabelEncoder()
        users[col + "_enc"] = le_knn.fit_transform(users[col].astype(str))

    for col, enc_col in [("Skin_Type", "Skin_Type_enc"), ("Hormonal_Status", "Hormonal_enc")]:
        le = LabelEncoder()
        users[enc_col] = le.fit_transform(users[col].astype(str))

    inter = pd.read_csv("interactions.csv", encoding="cp949",
                        usecols=[0, 1, 2], low_memory=False)
    inter.columns = ["User_ID", "Product_ID", "User_Rating"]
    inter = inter.dropna()
    inter["User_ID"] = inter["User_ID"].astype(int)
    inter["Product_ID"] = inter["Product_ID"].astype(int)
    inter["User_Rating"] = pd.to_numeric(inter["User_Rating"], errors="coerce")
    inter = inter.dropna(subset=["User_Rating"])

    y_labels = pd.read_csv("y_labels_final.csv")
    y_labels[SEVERITY_COLS] = y_labels[SEVERITY_COLS].fillna(0)

    products = pd.read_excel(excel_path, sheet_name="전체_통합DB")
    products.rename(columns={"Product ID": "Product_ID"}, inplace=True)

    max_oy = products["평균 평점"].max() or 1
    products["oy_norm"] = products["평균 평점"].fillna(0) / max_oy

    return users, inter, y_labels, products


# ─────────────────────────────────────────────
# 2. KNN 모델 구축
# ─────────────────────────────────────────────
@st.cache_resource
def build_knn(_users_df):
    """KNN 모델 구축 — Severity 5개 + Age + gender + Climate + Diet + Skin_Type + Hormonal"""
    feat_df = _users_df[KNN_FEATURE_COLS].fillna(0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(feat_df)
    knn = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
    knn.fit(X)
    return knn, scaler


# ─────────────────────────────────────────────
# 2-a. SVD 모델 로드
# ─────────────────────────────────────────────
SVD_PATH = "results/svd_model.pkl"

@st.cache_resource
def load_svd_model():
    if not os.path.exists(SVD_PATH):
        return None
    with open(SVD_PATH, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# 2-b. 피부 분석 딥러닝 모델 로드
# ─────────────────────────────────────────────
REGRESSOR_PATH  = "results/resnet50_final.pth"
CLASSIFIER_PATH = "results/sensitivity_cls_mobilenet_v3.pth"

_IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_skin_models():
    reg = tvmodels.resnet50()
    reg.fc = nn.Sequential(
        nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(256, 5)
    )
    reg.load_state_dict(
        torch.load(REGRESSOR_PATH, map_location="cpu", weights_only=False)
    )
    reg.eval()

    cls = tvmodels.mobilenet_v3_large()
    cls.classifier[3] = nn.Sequential(
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 2),
    )
    cls.load_state_dict(
        torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
    )
    cls.eval()
    return reg, cls


def infer_skin_scores(image_bytes: bytes) -> dict:
    reg, cls = load_skin_models()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _IMG_TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        reg_out = reg(x).squeeze().tolist()
        cls_pred = int(cls(x).argmax(dim=1).item())

    def _safe(v, lo, hi):
        v = float(v) if not (v != v) else 0.0  # NaN guard
        return float(np.clip(v, lo, hi))

    return {
        "Acne_Severity":         _safe(reg_out[0], 0.0, 10.0),
        "Dryness_AI":            _safe(reg_out[1], 0.0, 10.0),
        "Aging_Severity":        _safe(reg_out[2], 0.0,  4.2),
        "Pigmentation_Severity": _safe(reg_out[3], 0.0,  6.0),
        "Sensitivity_AI":        6.49 if cls_pred == 1 else 0.0,
    }


def classify_brightness(image_bytes: bytes) -> str:
    """HSV V채널(= RGB 최댓값) 평균으로 명도 판별.
    V < 80 → too_dark / V > 180 → too_bright / 그 외 → normal"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img, dtype=np.float32)
    mean_v = img_np.max(axis=2).mean()
    if mean_v < 80:
        return "too_dark"
    if mean_v > 180:
        return "too_bright"
    return "normal"


# ─────────────────────────────────────────────
# 2-c. BERT 상품명 임베딩
# ─────────────────────────────────────────────
@st.cache_resource
def build_product_embeddings(_products_df):
    """올리브영 상품명을 BERT로 임베딩하여 제품 표현 강화
    sentence-transformers 다국어 모델 사용 (한국어 지원)"""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    names = _products_df["올리브영 상품명"].fillna("").tolist()
    embeddings = model.encode(names, batch_size=32, show_progress_bar=False)
    pca = PCA(n_components=min(128, embeddings.shape[1]))
    embeddings_reduced = pca.fit_transform(embeddings)
    pid_to_emb = {
        int(pid): embeddings_reduced[i]
        for i, pid in enumerate(_products_df["Product_ID"])
    }
    return pid_to_emb, pca


# ─────────────────────────────────────────────
# 2-d. BPR 모델
# ─────────────────────────────────────────────
@st.cache_resource
def build_bpr(_inter_df, n_factors=128, n_epochs=100, lr=0.005, reg=0.001):
    """BPR 학습 — popularity-based negative sampling + 전체 유저 학습"""
    user_ids = _inter_df["User_ID"].unique()
    product_ids = _inter_df["Product_ID"].unique()
    user_idx = {u: i for i, u in enumerate(user_ids)}
    prod_idx = {p: i for i, p in enumerate(product_ids)}

    n_users = len(user_ids)
    n_items = len(product_ids)

    np.random.seed(42)
    U = np.random.normal(0, 0.1, (n_users, n_factors))
    V = np.random.normal(0, 0.1, (n_items, n_factors))

    user_pos = _inter_df.groupby("User_ID")["Product_ID"].apply(list).to_dict()
    all_pids = list(product_ids)

    # 인기도 기반 negative sampling — 인기 있는 제품을 negative로 써서 더 어려운 학습
    item_popularity = _inter_df["Product_ID"].value_counts().to_dict()
    pop_pids = [pid for pid, _ in sorted(item_popularity.items(),
                                          key=lambda x: x[1], reverse=True)
                if pid in prod_idx]
    pop_pids = pop_pids[:len(pop_pids)//2]  # 상위 50% 인기 제품을 negative pool로

    for epoch in range(n_epochs):
        for uid in np.random.choice(list(user_pos.keys()),
                                    len(user_pos), replace=False):
            if uid not in user_idx:
                continue
            u_i = user_idx[uid]
            pos_items = user_pos[uid]
            if not pos_items:
                continue
            pos_pid = np.random.choice(pos_items)
            if pos_pid not in prod_idx:
                continue
            p_i = prod_idx[pos_pid]
            # 70% 확률로 인기 제품에서 negative 샘플링 (더 어려운 학습)
            if np.random.random() < 0.7 and pop_pids:
                neg_pid = np.random.choice(pop_pids)
                tries = 0
                while neg_pid in pos_items and tries < 10:
                    neg_pid = np.random.choice(pop_pids)
                    tries += 1
                if neg_pid in pos_items:
                    neg_pid = np.random.choice(all_pids)
            else:
                neg_pid = np.random.choice(all_pids)
                while neg_pid in pos_items:
                    neg_pid = np.random.choice(all_pids)
            if neg_pid not in prod_idx:
                continue
            n_i = prod_idx[neg_pid]

            x_uij = np.dot(U[u_i], V[p_i] - V[n_i])
            sigmoid = 1 / (1 + np.exp(-x_uij))
            grad = 1 - sigmoid

            U[u_i] += lr * (grad * (V[p_i] - V[n_i]) - reg * U[u_i])
            V[p_i] += lr * (grad * U[u_i] - reg * V[p_i])
            V[n_i] += lr * (-grad * U[u_i] - reg * V[n_i])

    return {
        "U": U, "V": V,
        "user_ids": user_ids, "product_ids": product_ids,
        "user_idx": user_idx, "prod_idx": prod_idx,
    }


# ─────────────────────────────────────────────
# 2-e. LightGCN 모델
# ─────────────────────────────────────────────
@st.cache_resource
def build_lightgcn(_inter_df, n_factors=128, n_layers=4, n_epochs=100,
                   lr=0.001, reg=0.0001):
    """LightGCN 학습 — 대각 행렬 대신 element-wise 연산으로 메모리 절약"""
    from scipy.sparse import csr_matrix

    user_ids = _inter_df["User_ID"].unique()
    product_ids = _inter_df["Product_ID"].unique()
    user_idx = {u: i for i, u in enumerate(user_ids)}
    prod_idx = {p: i for i, p in enumerate(product_ids)}

    n_users = len(user_ids)
    n_items = len(product_ids)

    # 벡터화: iterrows 대신 map 사용
    valid_mask = (_inter_df["User_ID"].isin(user_idx)) & (_inter_df["Product_ID"].isin(prod_idx))
    valid_df = _inter_df[valid_mask]
    row_idx = valid_df["User_ID"].map(user_idx).to_numpy()
    col_idx = valid_df["Product_ID"].map(prod_idx).to_numpy()

    R = csr_matrix((np.ones(len(row_idx)), (row_idx, col_idx)),
                   shape=(n_users, n_items))

    d_u = np.array(R.sum(axis=1)).flatten() + 1e-10
    d_i = np.array(R.sum(axis=0)).flatten() + 1e-10
    # element-wise 정규화 — np.diag(15000×15000) 대신 브로드캐스팅으로 메모리 절약
    d_u_inv = (1.0 / np.sqrt(d_u))
    d_i_inv = (1.0 / np.sqrt(d_i))
    R_dense = R.toarray()
    A_norm = d_u_inv[:, None] * R_dense * d_i_inv[None, :]

    np.random.seed(42)
    E_u = np.random.normal(0, 0.1, (n_users, n_factors))
    E_i = np.random.normal(0, 0.1, (n_items, n_factors))

    # user_pos 벡터화
    user_pos = _inter_df[valid_mask].groupby("User_ID")["Product_ID"].apply(list).to_dict()
    all_pids = list(product_ids)

    item_popularity_lgcn = _inter_df[valid_mask]["Product_ID"].value_counts().to_dict()
    pop_pids_lgcn = [pid for pid, _ in sorted(item_popularity_lgcn.items(),
                                               key=lambda x: x[1], reverse=True)
                     if pid in prod_idx]
    pop_pids_lgcn = pop_pids_lgcn[:len(pop_pids_lgcn)//2]

    for epoch in range(n_epochs):
        E_u_agg = E_u.copy()
        E_i_agg = E_i.copy()
        E_u_cur, E_i_cur = E_u.copy(), E_i.copy()

        for layer in range(n_layers):
            E_u_new = A_norm @ E_i_cur
            E_i_new = A_norm.T @ E_u_cur
            E_u_agg += E_u_new
            E_i_agg += E_i_new
            E_u_cur = E_u_new
            E_i_cur = E_i_new

        E_u_final = E_u_agg / (n_layers + 1)
        E_i_final = E_i_agg / (n_layers + 1)

        sample_users = np.random.choice(
            list(user_pos.keys()), len(user_pos), replace=False
        )
        for uid in sample_users:
            if uid not in user_idx or not user_pos.get(uid):
                continue
            u_i = user_idx[uid]
            pos_pid = np.random.choice(user_pos[uid])
            if pos_pid not in prod_idx:
                continue
            p_i = prod_idx[pos_pid]
            if np.random.random() < 0.7 and pop_pids_lgcn:
                neg_pid = np.random.choice(pop_pids_lgcn)
                tries = 0
                while neg_pid in user_pos[uid] and tries < 10:
                    neg_pid = np.random.choice(pop_pids_lgcn)
                    tries += 1
                if neg_pid in user_pos[uid]:
                    neg_pid = np.random.choice(all_pids)
            else:
                neg_pid = np.random.choice(all_pids)
                while neg_pid in user_pos[uid]:
                    neg_pid = np.random.choice(all_pids)
            if neg_pid not in prod_idx:
                continue
            n_i = prod_idx[neg_pid]

            x_uij = (np.dot(E_u_final[u_i], E_i_final[p_i]) -
                     np.dot(E_u_final[u_i], E_i_final[n_i]))
            sigmoid = 1 / (1 + np.exp(-x_uij))
            grad = 1 - sigmoid

            E_u[u_i] += lr * (grad * (E_i_final[p_i] - E_i_final[n_i])
                               - reg * E_u[u_i])
            E_i[p_i] += lr * (grad * E_u_final[u_i] - reg * E_i[p_i])
            E_i[n_i] += lr * (-grad * E_u_final[u_i] - reg * E_i[n_i])

    return {
        "E_u": E_u_final, "E_i": E_i_final,
        "user_ids": user_ids, "product_ids": product_ids,
        "user_idx": user_idx, "prod_idx": prod_idx,
        "A_norm": A_norm,
    }


# ─────────────────────────────────────────────
# 3. Severity → 기능 가중치
# ─────────────────────────────────────────────
def get_user_weights(user_scores: dict) -> dict:
    weights: dict = {}
    for col, funcs in SEVERITY_FUNCTION_MAP.items():
        w = max(0.0, float(user_scores.get(col, 0))) / 10.0
        for f in funcs:
            weights[f] = weights.get(f, 0) + w
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def get_baumann_type(user_scores: dict) -> str:
    """Baumann 16 타입 분류 (Baumann 2008 기준치 적용).
    D/O: Dryness≥2.88 → D, else O
    S/R: Acne≥3.94 OR Sensitivity≥1.45 → S, else R
    P/N: Pigmentation≥1.97 → P, else N
    W/T: Aging≥1.02 → W, else T"""
    dry  = user_scores.get("Dryness_Severity", 0)
    acne = user_scores.get("Acne_Severity", 0)
    sens = user_scores.get("Sensitivity_Severity", 0)
    pig  = user_scores.get("Pigmentation_Severity", 0)
    age  = user_scores.get("Aging_Severity", 0)

    d_axis = "D" if dry  >= 2.88 else "O"
    s_axis = "S" if (acne >= 3.94 or sens >= 1.45) else "R"
    p_axis = "P" if pig  >= 1.97 else "N"
    w_axis = "W" if age  >= 1.02 else "T"
    return f"{d_axis}{s_axis}{p_axis}{w_axis}"


def feature_match_score(func_str, user_weights: dict) -> float:
    if pd.isna(func_str):
        return 0.0
    return sum(user_weights.get(f.strip(), 0) for f in str(func_str).split("|"))


# ─────────────────────────────────────────────
# 4. 동적 가중치
# ─────────────────────────────────────────────
def get_dynamic_weights(user_scores: dict) -> tuple:
    """Severity 기반 동적 가중치 계산
    피부 고민이 명확할수록 기능매칭(CB) 비중 자동으로 높아짐
    피부 고민이 약할수록 KNN 비중 높아짐"""
    severity_vals = [user_scores.get(c, 0) for c in SEVERITY_COLS]
    mean_sev = np.mean(severity_vals) if severity_vals else 0

    alpha = 0.3 + 0.3 * (mean_sev / 10.0)
    beta  = 0.5 - 0.2 * (mean_sev / 10.0)
    gamma = 0.2

    total = alpha + beta + gamma
    return round(alpha / total, 3), round(beta / total, 3), round(gamma / total, 3)


# ─────────────────────────────────────────────
# 5. 협업 필터링 점수 함수
# ─────────────────────────────────────────────
def _get_feat_vals(user_scores: dict) -> list:
    fv = []
    for col in KNN_FEATURE_COLS:
        if col in user_scores:
            fv.append(float(user_scores[col]))
        elif col == "Age":
            fv.append(float(user_scores.get("age_input", 30)))
        elif col == "gender":
            fv.append(float(user_scores.get("gender_input", 0)))
        elif col == "Climate_enc":
            fv.append(float(user_scores.get("climate_enc", 0)))
        elif col == "Diet_enc":
            fv.append(float(user_scores.get("diet_enc", 0)))
        elif col == "Skin_Type_enc":
            fv.append(float(user_scores.get("skin_type_enc", 0)))
        elif col == "Hormonal_enc":
            fv.append(float(user_scores.get("hormonal_enc", 0)))
        else:
            fv.append(0.0)
    return fv


def knn_product_scores(user_scores: dict, users_df, knn_mdl, scaler, inter_df) -> dict:
    feat_sc = scaler.transform([_get_feat_vals(user_scores)])
    _, idx = knn_mdl.kneighbors(feat_sc)
    neighbor_ids = users_df.iloc[idx[0]]["User_ID"].tolist()
    nb_inter = inter_df[inter_df["User_ID"].isin(neighbor_ids)]
    return nb_inter.groupby("Product_ID")["User_Rating"].mean().to_dict()


def svd_product_scores(user_scores: dict, users_df, svd_model: dict,
                       knn_mdl, scaler) -> dict:
    if svd_model is None:
        return {}
    feat_sc = scaler.transform([_get_feat_vals(user_scores)])
    _, idx = knn_mdl.kneighbors(feat_sc)
    neighbor_ids = users_df.iloc[idx[0]]["User_ID"].tolist()
    scores = {}
    valid_count = 0
    for uid in neighbor_ids:
        if uid not in svd_model["user_idx"]:
            continue
        u_i = svd_model["user_idx"][uid]
        valid_count += 1
        for pid, p_i in svd_model["prod_idx"].items():
            pred = float(svd_model["pred_matrix"][u_i, p_i])
            scores[pid] = scores.get(pid, 0.0) + pred
    if valid_count > 0:
        scores = {pid: v / valid_count for pid, v in scores.items()}
    return scores


def bpr_product_scores(user_scores: dict, users_df, bpr_model: dict,
                       knn_mdl, scaler) -> dict:
    if bpr_model is None:
        return {}
    feat_sc = scaler.transform([_get_feat_vals(user_scores)])
    _, idx = knn_mdl.kneighbors(feat_sc)
    neighbor_ids = users_df.iloc[idx[0]]["User_ID"].tolist()
    scores = {}
    valid_count = 0
    for uid in neighbor_ids:
        if uid not in bpr_model["user_idx"]:
            continue
        u_i = bpr_model["user_idx"][uid]
        valid_count += 1
        for pid, p_i in bpr_model["prod_idx"].items():
            pred = float(np.dot(bpr_model["U"][u_i], bpr_model["V"][p_i]))
            scores[pid] = scores.get(pid, 0.0) + pred
    if valid_count > 0:
        scores = {pid: v / valid_count for pid, v in scores.items()}
    return scores


def lightgcn_product_scores(user_scores: dict, users_df, lgcn_model: dict,
                             knn_mdl, scaler) -> dict:
    if lgcn_model is None:
        return {}
    feat_sc = scaler.transform([_get_feat_vals(user_scores)])
    _, idx = knn_mdl.kneighbors(feat_sc)
    neighbor_ids = users_df.iloc[idx[0]]["User_ID"].tolist()
    scores = {}
    valid_count = 0
    for uid in neighbor_ids:
        if uid not in lgcn_model["user_idx"]:
            continue
        u_i = lgcn_model["user_idx"][uid]
        valid_count += 1
        for pid, p_i in lgcn_model["prod_idx"].items():
            pred = float(np.dot(lgcn_model["E_u"][u_i], lgcn_model["E_i"][p_i]))
            scores[pid] = scores.get(pid, 0.0) + pred
    if valid_count > 0:
        scores = {pid: v / valid_count for pid, v in scores.items()}
    return scores


def _minmax_norm(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-9)


# ─────────────────────────────────────────────
# 6. 하이브리드 추천 엔진
# ─────────────────────────────────────────────
def recommend(
    user_scores: dict,
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    knn_mdl,
    scaler,
    inter_df: pd.DataFrame,
    alpha: float = None,
    beta: float = None,
    gamma: float = 0.2,
    model_type: str = "hybrid",
    top_candidates: int = 3,
    svd_model=None,
    bpr_model=None,
    lgcn_model=None,
    pid_to_emb=None,
):
    """카테고리별 상위 top_candidates 후보 반환"""
    user_weights = get_user_weights(user_scores)

    if alpha is None or beta is None:
        alpha, beta, gamma = get_dynamic_weights(user_scores)

    df = products_df.copy()

    # 기능 매칭 점수
    df["feat_raw"] = df["기능(Function)"].apply(
        lambda x: feature_match_score(x, user_weights))
    max_feat = df["feat_raw"].max() or 1.0
    df["feat_score"] = df["feat_raw"] / max_feat

    # BERT 임베딩 — 사용자 Severity 기반 쿼리 벡터와 제품 임베딩 코사인 유사도
    if pid_to_emb is not None and len(pid_to_emb) > 0:
        user_weights_for_bert = get_user_weights(user_scores)
        top3_funcs = sorted(user_weights_for_bert.items(),
                            key=lambda x: x[1], reverse=True)[:3]

        query_embs = []
        for func, weight in top3_funcs:
            func_products = df[df["기능(Function)"].str.contains(func, na=False)]
            for pid in func_products["Product_ID"]:
                if int(pid) in pid_to_emb:
                    query_embs.append(pid_to_emb[int(pid)] * weight)

        if query_embs:
            query_vec = np.mean(query_embs, axis=0)
            query_norm = np.linalg.norm(query_vec)

            bert_scores = []
            for pid in df["Product_ID"]:
                if int(pid) in pid_to_emb:
                    prod_emb = pid_to_emb[int(pid)]
                    prod_norm = np.linalg.norm(prod_emb)
                    if query_norm > 0 and prod_norm > 0:
                        cosine_sim = float(np.dot(query_vec, prod_emb) /
                                          (query_norm * prod_norm))
                        bert_scores.append((cosine_sim + 1) / 2)
                    else:
                        bert_scores.append(0.5)
                else:
                    bert_scores.append(0.5)

            df["bert_score"] = bert_scores
            df["feat_score"] = 0.6 * df["feat_score"] + 0.4 * df["bert_score"]
        else:
            df["bert_score"] = 0.5
    else:
        df["bert_score"] = 0.0

    # KNN 점수
    if model_type in ("knn", "hybrid"):
        knn_sc = knn_product_scores(user_scores, users_df, knn_mdl, scaler, inter_df)
        df["knn_score"] = df["Product_ID"].map(knn_sc).fillna(0) / 5.0
    else:
        df["knn_score"] = 0.0

    # SVD 점수
    if model_type in ("svd", "hybrid_svd") and svd_model is not None:
        svd_sc = svd_product_scores(user_scores, users_df, svd_model, knn_mdl, scaler)
        if svd_sc:
            df["svd_score"] = _minmax_norm(df["Product_ID"].map(svd_sc).fillna(0))
        else:
            df["svd_score"] = 0.0
    else:
        df["svd_score"] = 0.0

    # BPR 점수
    if model_type in ("bpr", "hybrid_bpr") and bpr_model is not None:
        bpr_sc = bpr_product_scores(user_scores, users_df, bpr_model, knn_mdl, scaler)
        if bpr_sc:
            df["bpr_score"] = _minmax_norm(df["Product_ID"].map(bpr_sc).fillna(0))
        else:
            df["bpr_score"] = 0.0
    else:
        df["bpr_score"] = 0.0

    # LightGCN 점수
    if model_type in ("lgcn", "hybrid_lgcn") and lgcn_model is not None:
        lgcn_sc = lightgcn_product_scores(user_scores, users_df, lgcn_model, knn_mdl, scaler)
        if lgcn_sc:
            df["lgcn_score"] = _minmax_norm(df["Product_ID"].map(lgcn_sc).fillna(0))
        else:
            df["lgcn_score"] = 0.0
    else:
        df["lgcn_score"] = 0.0

    # 최종 점수
    if model_type == "content":
        df["final_score"] = df["feat_score"]
    elif model_type == "knn":
        df["final_score"] = df["knn_score"]
    elif model_type == "svd":
        df["final_score"] = df["svd_score"]
    elif model_type == "bpr":
        df["final_score"] = df["bpr_score"]
    elif model_type == "lgcn":
        df["final_score"] = df["lgcn_score"]
    elif model_type == "hybrid_svd":
        df["final_score"] = alpha * df["feat_score"] + beta * df["svd_score"] + gamma * df["oy_norm"]
    elif model_type == "hybrid_bpr":
        df["final_score"] = alpha * df["feat_score"] + beta * df["bpr_score"] + gamma * df["oy_norm"]
    elif model_type == "hybrid_lgcn":
        df["final_score"] = alpha * df["feat_score"] + beta * df["lgcn_score"] + gamma * df["oy_norm"]
    elif model_type == "ensemble":
        knn_sc_e = knn_product_scores(user_scores, users_df, knn_mdl, scaler, inter_df)
        df["knn_score_e"] = _minmax_norm(df["Product_ID"].map(knn_sc_e).fillna(0) / 5.0)

        bpr_sc_e = bpr_product_scores(user_scores, users_df, bpr_model, knn_mdl, scaler) if bpr_model else {}
        df["bpr_score_e"] = _minmax_norm(df["Product_ID"].map(bpr_sc_e).fillna(0)) if bpr_sc_e else 0.0

        lgcn_sc_e = lightgcn_product_scores(user_scores, users_df, lgcn_model, knn_mdl, scaler) if lgcn_model else {}
        df["lgcn_score_e"] = _minmax_norm(df["Product_ID"].map(lgcn_sc_e).fillna(0)) if lgcn_sc_e else 0.0

        df["cf_ensemble"] = (df["knn_score_e"] + df["bpr_score_e"] + df["lgcn_score_e"]) / 3.0
        df["final_score"] = alpha * df["feat_score"] + beta * df["cf_ensemble"] + gamma * df["oy_norm"]
    elif model_type == "hybrid_ensemble":
        knn_sc_he = knn_product_scores(user_scores, users_df, knn_mdl, scaler, inter_df)
        df["knn_score_he"] = _minmax_norm(df["Product_ID"].map(knn_sc_he).fillna(0) / 5.0)

        bpr_sc_he = bpr_product_scores(user_scores, users_df, bpr_model, knn_mdl, scaler) if bpr_model else {}
        df["bpr_score_he"] = _minmax_norm(df["Product_ID"].map(bpr_sc_he).fillna(0)) if bpr_sc_he else 0.0

        lgcn_sc_he = lightgcn_product_scores(user_scores, users_df, lgcn_model, knn_mdl, scaler) if lgcn_model else {}
        df["lgcn_score_he"] = _minmax_norm(df["Product_ID"].map(lgcn_sc_he).fillna(0)) if lgcn_sc_he else 0.0

        df["cf_ensemble_he"] = (df["knn_score_he"] + df["bpr_score_he"] + df["lgcn_score_he"]) / 3.0
        df["final_score"] = alpha * df["feat_score"] + beta * df["cf_ensemble_he"] + gamma * df["oy_norm"]
    else:  # hybrid (KNN)
        df["final_score"] = alpha * df["feat_score"] + beta * df["knn_score"] + gamma * df["oy_norm"]

    recs = []
    for cat in CATEGORIES:
        cat_df = df[df["카테고리"] == cat]
        if cat_df.empty:
            continue
        candidates = cat_df.nlargest(top_candidates, "final_score")
        for _, best in candidates.iterrows():
            recs.append({
                "category":    cat,
                "product_id":  int(best["Product_ID"]),
                "name":        str(best["올리브영 상품명"]),
                "brand":       str(best["올리브영 브랜드"]),
                "functions":   str(best["기능(Function)"]),
                "feat_score":  round(float(best["feat_score"]), 4),
                "knn_score":   round(float(best.get("knn_score", 0)), 4),
                "oy_score":    round(float(best["oy_norm"]), 4),
                "final_score": round(float(best["final_score"]), 4),
                "price":       best["할인가(원)"],
                "oy_rank":     best["올리브영 순위"],
                "avg_rating":  best["평균 평점"],
            })
    return recs, df


# ─────────────────────────────────────────────
# 7. Leave-one-out 평가
# ─────────────────────────────────────────────
def run_evaluation(users_df, inter_df, products_df, knn_mdl, scaler,
                   n_eval: int = 200, k_list=(5, 10, 15),
                   svd_model=None, bpr_model=None, lgcn_model=None):
    """Content-based / KNN / Hybrid / SVD / BPR / LightGCN 모델 비교"""
    valid_pids = set(products_df["Product_ID"].tolist())
    inter_v = inter_df[inter_df["Product_ID"].isin(valid_pids)].copy()

    cnt = inter_v.groupby("User_ID").size()
    eligible = cnt[cnt >= 2].index.tolist()
    np.random.seed(42)
    sample = np.random.choice(eligible, min(n_eval, len(eligible)), replace=False)

    models = ["content", "knn", "hybrid", "svd", "ensemble"]

    hits    = {m: {k: 0   for k in k_list} for m in models}
    ndcg    = {m: {k: 0.0 for k in k_list} for m in models}
    total   = {m: 0 for m in models}
    cat_cov = {m: set() for m in models}

    users_idx = users_df.set_index("User_ID")
    pbar = st.progress(0, text="평가 진행 중...")

    for i, uid in enumerate(sample):
        pbar.progress((i + 1) / len(sample),
                      text=f"평가 중 {i+1}/{len(sample)} | 유저 {uid}")
        if uid not in users_idx.index:
            continue

        u_inter = inter_v[inter_v["User_ID"] == uid]
        held = u_inter.nlargest(1, "User_Rating").iloc[0]
        held_pid = int(held["Product_ID"])

        u_row = users_idx.loc[uid]
        u_scores = {c: float(u_row.get(c, 0) or 0) for c in SEVERITY_COLS}
        u_scores["age_input"]      = float(u_row.get("Age", 30) or 30)
        u_scores["gender_input"]   = float(u_row.get("gender", 0) or 0)
        u_scores["climate_enc"]    = float(u_row.get("Climate_enc", 0) or 0)
        u_scores["diet_enc"]       = float(u_row.get("Diet_enc", 0) or 0)
        u_scores["skin_type_enc"]  = float(u_row.get("Skin_Type_enc", 0) or 0)
        u_scores["hormonal_enc"]   = float(u_row.get("Hormonal_enc", 0) or 0)

        for model in models:
            recs, score_df = recommend(
                u_scores, products_df, users_df, knn_mdl, scaler, inter_df,
                model_type=model,
                svd_model=svd_model,
                bpr_model=bpr_model,
                lgcn_model=lgcn_model,
            )
            all_ranked = score_df.nlargest(max(k_list), "final_score")["Product_ID"].tolist()

            total[model] += 1
            for k in k_list:
                top_k = all_ranked[:k]
                if held_pid in top_k:
                    hits[model][k] += 1
                    rank = top_k.index(held_pid) + 1
                    ndcg[model][k] += 1.0 / np.log2(rank + 1)

            cat_cov[model].update(r["category"] for r in recs)

    pbar.empty()

    rows = []
    for m in models:
        t = total[m] or 1
        row = {"모델": MODEL_LABELS.get(m, m)}
        for k in k_list:
            row[f"Precision@{k}"] = round(hits[m][k] / t, 4)
            row[f"NDCG@{k}"]      = round(ndcg[m][k] / t, 4)
        row["Category Coverage"] = round(len(cat_cov[m]) / len(CATEGORIES), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 8. UI 컴포넌트
# ─────────────────────────────────────────────
def render_card(rec: dict):
    funcs = [f.strip() for f in str(rec["functions"]).split("|") if f.strip()]
    tags = "".join(
        f'<span class="func-tag">{FUNCTION_KOR.get(f, f)}</span>' for f in funcs
    )
    color = CAT_COLOR.get(rec["category"], "#ccc")
    price_str  = f"{int(rec['price']):,}원"    if pd.notna(rec["price"])      else "-"
    rank_str   = f"#{int(rec['oy_rank'])}위"   if pd.notna(rec["oy_rank"])    else "-"
    rating_str = f"{rec['avg_rating']:.2f}"    if pd.notna(rec["avg_rating"]) else "-"

    def bar(val):
        pct = int(val * 100)
        return (
            f'<div style="background:#e9ecef;border-radius:4px;height:6px;margin-top:2px">'
            f'<div style="background:{color};width:{pct}%;height:6px;border-radius:4px"></div>'
            f'</div>'
        )

    st.markdown(f"""
<div class="rec-card" style="border-left-color:{color}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <span style="background:{color};color:white;padding:2px 10px;
          border-radius:10px;font-size:12px;font-weight:bold">
      {CAT_DISPLAY.get(rec['category'], rec['category'])}
    </span>
    <span style="font-size:16px;font-weight:bold;color:{color}">
      추천 점수 {rec['final_score']:.3f}
    </span>
  </div>
  <div style="font-weight:bold;font-size:15px;margin-bottom:3px">{rec['name']}</div>
  <div style="color:#666;font-size:13px;margin-bottom:8px">
    {rec['brand']} &nbsp;|&nbsp; {price_str} &nbsp;|&nbsp; 올리브영 {rank_str}
    &nbsp;|&nbsp; 평점 {rating_str}
  </div>
  <div style="margin-bottom:10px">{tags}</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:12px">
    <div>
      <span class="score-label">기능매칭 (α, 동적)</span>
      <div style="font-weight:bold">{rec['feat_score']:.3f}</div>
      {bar(rec['feat_score'])}
    </div>
    <div>
      <span class="score-label">협업필터링 (β, 동적)</span>
      <div style="font-weight:bold">{rec['knn_score']:.3f}</div>
      {bar(rec['knn_score'])}
    </div>
    <div>
      <span class="score-label">올리브영평점 (γ=0.2)</span>
      <div style="font-weight:bold">{rec['oy_score']:.3f}</div>
      {bar(rec['oy_score'])}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_baumann_card(baumann_type: str):
    data = BAUMANN_TYPE_DATA.get(baumann_type)
    if data is None:
        return

    ind_tags = ""
    for label, is_pos in data["indicators"].items():
        color = "#FF6B6B" if is_pos else "#4ECDC4"
        ind_tags += (
            f'<span style="display:inline-block;background:{color};color:white;'
            f'border-radius:8px;padding:2px 10px;margin:3px 2px;font-size:12px">'
            f'{label}</span>'
        )

    st.markdown(f"""
<div style="background:#f0f7ff;border-radius:12px;padding:20px 24px;
            margin-top:24px;border:1px solid #bee3f8">
  <div style="font-size:12px;color:#888;margin-bottom:4px">Baumann 피부 타입</div>
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:6px">
    <span style="font-size:32px;font-weight:bold;color:#2c7da0">{baumann_type}</span>
    <span style="font-size:15px;font-weight:600;color:#444">{data['full_name']}</span>
  </div>
  <div style="margin-bottom:10px">{ind_tags}</div>
  <div style="color:#333;font-size:13px;margin-bottom:12px">{data['description']}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
    <div style="background:#fff;border-radius:8px;padding:10px 14px">
      <div style="font-size:12px;font-weight:700;color:#555;margin-bottom:6px">주요 고민</div>
      <div style="font-size:12px;font-weight:400;color:#222;line-height:1.6">{data['main_concern']}</div>
    </div>
    <div style="background:#fff;border-radius:8px;padding:10px 14px">
      <div style="font-size:12px;font-weight:700;color:#555;margin-bottom:6px">케어 방향</div>
      <div style="font-size:12px;font-weight:400;color:#222;line-height:1.6">{data['care_direction']}</div>
    </div>
  </div>
  <div style="font-size:11px;color:#bbb">
    출처: Baumann, L. (2008). Dermatologic Clinics, 26(3), 359–373.
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 9. MAIN APP
# ─────────────────────────────────────────────
def main():
    st.title("얼굴 이미지 기반 스킨케어 루틴 추천")

    with st.spinner("데이터 및 모델 로딩 중... (BPR·LightGCN 학습 포함)"):
        users, inter, y_labels, products = load_data()
        knn_mdl, scaler = build_knn(users)
        svd_mdl = load_svd_model()
        pid_to_emb, _ = build_product_embeddings(products)

        try:
            bpr_mdl = build_bpr(inter)
        except Exception as e:
            st.error(f"❌ BPR 빌드 실패: {e}")
            bpr_mdl = None

        try:
            lgcn_mdl = build_lightgcn(inter)
        except Exception as e:
            st.error(f"❌ LightGCN 빌드 실패: {e}")
            lgcn_mdl = None

    tab_rec, tab_eval, tab_data = st.tabs(
        ["🎯 추천받기", "📊 성능 비교", "📈 EDA / 데이터 분석"]
    )

    # ══════════════════════════════════════
    # TAB 1 : 추천받기
    # ══════════════════════════════════════
    with tab_rec:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.subheader("피부 상태 입력")
            st.markdown("**📋 피부 상태 입력 방법을 선택하세요**")
            with st.expander("💡 입력 방법 안내"):
                st.markdown("""
- **📷 얼굴 사진으로 분석**: 사진을 업로드하면 AI가 여드름·노화·색소침착을 자동 분석합니다. 건조함과 민감성은 간단한 설문으로 보완합니다.
- **슬라이더 직접 입력**: 피부 고민 정도를 직접 조절합니다. 0은 고민 없음, 10은 매우 심각합니다.

> **Severity란?** 피부 상태의 심각도를 0~10 사이 수치로 표현한 것입니다. 값이 높을수록 해당 피부 고민이 심한 것을 의미합니다.
                """)
            input_mode = st.radio(
                "입력 방법",
                ["슬라이더 직접 입력", "📷 얼굴 사진으로 분석"],
                horizontal=True,
            )

            SLIDER_HELP = {
                "Acne_Severity":         "여드름, 뾰루지, 트러블이 얼마나 자주/심하게 나는지 (0: 없음 / 5: 가끔 / 10: 매우 심함)",
                "Dryness_Severity":      "세안 후 또는 낮 동안 피부가 당기거나 건조한 정도 (0: 촉촉함 / 5: 가끔 당김 / 10: 항상 건조)",
                "Pigmentation_Severity": "기미, 잡티, 색소침착의 정도 (0: 없음 / 3: 중간 / 6: 매우 심함)",
                "Aging_Severity":        "주름, 탄력 저하, 처짐의 정도 (0: 없음 / 2: 중간 / 4.2: 매우 심함)",
            }

            if input_mode == "슬라이더 직접 입력":
                user_scores: dict = {}
                for col in SEVERITY_COLS:
                    emoji = SEVERITY_EMOJI[col]
                    label = SEVERITY_LABELS[col]
                    if col == "Sensitivity_Severity":
                        sens = st.radio(
                            f"{emoji} {label}",
                            ["비민감 (0.0)", "민감 (6.49)"],
                            horizontal=True,
                            help="민감성 피부 여부를 선택하세요",
                        )
                        user_scores[col] = 6.49 if "민감" in sens and "비민감" not in sens else 0.0
                    else:
                        user_scores[col] = st.slider(
                            f"{emoji} {label}", 0.0, 10.0, 3.0, 0.1,
                            help=SLIDER_HELP.get(col, "0: 없음 · 5: 보통 · 10: 심각"),
                        )

                st.divider()
                st.markdown("**👤 기본 정보**")
                col_g2, col_a2 = st.columns(2)
                with col_g2:
                    gender_opt2 = st.radio("성별", ["여성 (0)", "남성 (1)"], horizontal=True, key="slider_gender",
                                           help="성별에 따라 유사 사용자 탐색 정확도가 높아집니다")
                    user_scores["gender_input"] = 0.0 if "여성" in gender_opt2 else 1.0
                with col_a2:
                    age_val2 = st.slider("연령", 1, 100, 30, 1, key="slider_age",
                                         help="실제 나이를 입력하세요. 추천 시 15~49세 범위로 자동 조정됩니다")
                    user_scores["age_input"] = float(max(15, min(age_val2, 49)))
                col_c2, col_d2 = st.columns(2)
                with col_c2:
                    climate_opt2 = st.selectbox("거주 기후", ["summer", "Temperate", "winter", "Dry"], key="slider_climate",
                                                help="summer: 고온다습 / Temperate: 온대 / winter: 한랭 / Dry: 건조")
                    climate_enc_map2 = {"summer": 2, "Temperate": 1, "winter": 3, "Dry": 0}
                    user_scores["climate_enc"] = float(climate_enc_map2[climate_opt2])
                with col_d2:
                    diet_opt2 = st.selectbox("식단 유형", ["Balanced", "Vegan", "High_Dairy", "Junk_Food", "High_Sugar"], key="slider_diet",
                                             help="Balanced: 균형식 / Vegan: 채식 / High_Dairy: 유제품 많음 / Junk_Food: 인스턴트 / High_Sugar: 당분 많음")
                    diet_enc_map2 = {"Balanced": 0, "Vegan": 4, "High_Dairy": 1, "Junk_Food": 2, "High_Sugar": 3}
                    user_scores["diet_enc"] = float(diet_enc_map2[diet_opt2])

                col_st, col_hs = st.columns(2)
                with col_st:
                    skin_type_opt = st.selectbox(
                        "피부 타입", ["Combination", "Dry", "Normal", "Oily", "Sensitive"],
                        key="slider_skin_type",
                        help="Combination: 복합성 / Dry: 건성 / Normal: 중성 / Oily: 지성 / Sensitive: 민감성"
                    )
                    skin_type_enc_map = {"Combination": 0, "Dry": 1, "Normal": 2, "Oily": 3, "Sensitive": 4}
                    user_scores["skin_type_enc"] = float(skin_type_enc_map[skin_type_opt])
                with col_hs:
                    hormonal_opt = st.selectbox(
                        "호르몬 상태", ["Stable", "Teen", "Pregnant", "PCOS"],
                        key="slider_hormonal",
                        help="Stable: 안정 / Teen: 10대 / Pregnant: 임신 중 / PCOS: 다낭성난소증후군"
                    )
                    hormonal_enc_map = {"Stable": 3, "Teen": 4, "Pregnant": 1, "PCOS": 2}
                    user_scores["hormonal_enc"] = float(hormonal_enc_map[hormonal_opt])

            elif input_mode == "📷 얼굴 사진으로 분석":
                uploaded = st.file_uploader(
                    "얼굴 사진 업로드 (jpg / png)",
                    type=["jpg", "jpeg", "png"],
                )

                if uploaded is not None:
                    st.image(uploaded, caption="업로드된 사진", use_container_width=True)

                    image_bytes = uploaded.read()
                    _brightness = classify_brightness(image_bytes)

                    if _brightness == "too_dark":
                        st.warning("⚠️ 이미지가 너무 어둡습니다 (HSV V채널 평균 < 80). 더 밝은 환경에서 재촬영 후 다시 업로드해 주세요.")
                        user_scores = {c: 0.0 for c in SEVERITY_COLS}
                    elif _brightness == "too_bright":
                        st.warning("⚠️ 이미지가 너무 밝습니다 (HSV V채널 평균 > 180). 직사광선을 피해 재촬영 후 다시 업로드해 주세요.")
                        user_scores = {c: 0.0 for c in SEVERITY_COLS}
                    else:
                        with st.spinner("AI 피부 분석 중..."):
                            inferred = infer_skin_scores(image_bytes)

                        st.success("분석 완료!")
                        st.divider()

                        st.markdown("**🤖 AI 분석 결과**")
                        auto_cols = ["Acne_Severity", "Aging_Severity", "Pigmentation_Severity"]
                        for col in auto_cols:
                            val = inferred[col]
                            bar_w = max(0, min(10, int(val)))
                            st.markdown(
                                f"**{SEVERITY_EMOJI[col]} {SEVERITY_LABELS[col]}** `{val:.2f}` "
                                f"{'█' * bar_w}{'░' * (10 - bar_w)}"
                            )

                        st.divider()
                        st.markdown("**📋 설문 (AI 보완용)**")

                        dryness_opt = st.radio(
                            "💧 피부 건조함이 어느 정도인가요?",
                            ["거의 없음", "가끔 당김", "자주 당김", "항상 건조"],
                            horizontal=True,
                        )
                        dryness_survey_map = {
                            "거의 없음": 1.0, "가끔 당김": 3.5,
                            "자주 당김": 6.5, "항상 건조": 9.0,
                        }
                        dryness_survey = dryness_survey_map[dryness_opt]
                        dryness_final = round(0.4 * inferred["Dryness_AI"] + 0.6 * dryness_survey, 2)

                        aging_opt = st.radio(
                            "⏳ 노화/주름 고민이 어느 정도인가요?",
                            ["거의 없음", "가끔 보임", "주름/처짐 있음", "심한 편"],
                            horizontal=True,
                        )
                        aging_survey_map = {
                            "거의 없음": 0.2, "가끔 보임": 1.0,
                            "주름/처짐 있음": 2.5, "심한 편": 4.0,
                        }
                        aging_survey = aging_survey_map[aging_opt]
                        aging_ai = inferred["Aging_Severity"]
                        aging_final = round(
                            aging_ai * 0.4 + aging_survey * 0.6 if aging_ai > 0.1
                            else aging_survey,
                            2
                        )

                        pig_opt = st.radio(
                            "⭐ 색소침착/기미 고민이 어느 정도인가요?",
                            ["거의 없음", "약간 있음", "잡티/기미 있음", "심한 편"],
                            horizontal=True,
                        )
                        pig_survey_map = {
                            "거의 없음": 0.3, "약간 있음": 1.5,
                            "잡티/기미 있음": 3.0, "심한 편": 5.5,
                        }
                        pig_survey = pig_survey_map[pig_opt]
                        pig_ai = inferred["Pigmentation_Severity"]
                        pig_final = round(
                            pig_ai * 0.4 + pig_survey * 0.6 if pig_ai > 0.1
                            else pig_survey,
                            2
                        )

                        sens_opt = st.radio(
                            "🌿 민감성 피부인가요?",
                            ["아니오", "예"],
                            horizontal=True,
                        )
                        sens_survey = 6.49 if sens_opt == "예" else 0.0
                        sens_ai = inferred["Sensitivity_AI"]
                        if sens_ai > 0 and sens_survey > 0:
                            sens_final = 6.49
                        elif sens_ai > 0 or sens_survey > 0:
                            sens_final = 3.25
                        else:
                            sens_final = 0.0

                        st.divider()
                        st.markdown("**👤 기본 정보 (추천 정확도 향상)**")
                        col_g, col_a = st.columns(2)
                        with col_g:
                            gender_opt = st.radio("성별", ["여성 (0)", "남성 (1)"], horizontal=True)
                            gender_val = 0.0 if "여성" in gender_opt else 1.0
                        with col_a:
                            age_val = st.slider("연령", 1, 100, 30, 1)
                            age_knn = max(15, min(age_val, 49))
                        col_c, col_d = st.columns(2)
                        with col_c:
                            climate_opt = st.selectbox("거주 기후", ["summer", "Temperate", "winter", "Dry"])
                            climate_enc_map = {"summer": 2, "Temperate": 1, "winter": 3, "Dry": 0}
                            climate_enc_val = climate_enc_map[climate_opt]
                        with col_d:
                            diet_opt = st.selectbox("식단 유형", ["Balanced", "Vegan", "High_Dairy", "Junk_Food", "High_Sugar"])
                            diet_enc_map = {"Balanced": 0, "Vegan": 4, "High_Dairy": 1, "Junk_Food": 2, "High_Sugar": 3}
                            diet_enc_val = diet_enc_map[diet_opt]

                        col_st2, col_hs2 = st.columns(2)
                        with col_st2:
                            skin_type_opt2 = st.selectbox(
                                "피부 타입", ["Combination", "Dry", "Normal", "Oily", "Sensitive"],
                                key="photo_skin_type"
                            )
                            skin_type_enc_map2 = {"Combination": 0, "Dry": 1, "Normal": 2, "Oily": 3, "Sensitive": 4}
                            skin_type_enc_val = float(skin_type_enc_map2[skin_type_opt2])
                        with col_hs2:
                            hormonal_opt2 = st.selectbox(
                                "호르몬 상태", ["Stable", "Teen", "Pregnant", "PCOS"],
                                key="photo_hormonal"
                            )
                            hormonal_enc_map2 = {"Stable": 3, "Teen": 4, "Pregnant": 1, "PCOS": 2}
                            hormonal_enc_val = float(hormonal_enc_map2[hormonal_opt2])

                        user_scores = {
                            "Acne_Severity":         inferred["Acne_Severity"],
                            "Dryness_Severity":      dryness_final,
                            "Aging_Severity":        aging_final,
                            "Pigmentation_Severity": pig_final,
                            "Sensitivity_Severity":  sens_final,
                            "age_input":             float(age_knn),
                            "gender_input":          gender_val,
                            "climate_enc":           float(climate_enc_val),
                            "diet_enc":              float(diet_enc_val),
                            "skin_type_enc":         skin_type_enc_val,
                            "hormonal_enc":          hormonal_enc_val,
                        }

                        st.divider()
                        st.markdown("**✅ 최종 입력값 요약**")
                        for col in SEVERITY_COLS:
                            val = user_scores[col]
                            bar_w = max(0, min(10, int(val)))
                            st.markdown(
                                f"**{SEVERITY_EMOJI[col]} {SEVERITY_LABELS[col]}** "
                                f"`{val:.2f}` {'█' * bar_w}{'░' * (10 - bar_w)}"
                            )
                        st.caption(
                            f"👤 성별: {'여성' if gender_val == 0 else '남성'} | "
                            f"연령: {age_val}세 | 기후: {climate_opt} | 식단: {diet_opt}"
                        )
                else:
                    st.info(
                        "jpg 또는 png 파일을 업로드하면 AI가 자동으로 피부를 분석하고, "
                        "건조함·민감성은 간단한 설문으로 입력할 수 있습니다."
                    )
                    user_scores = {c: 0.0 for c in SEVERITY_COLS}

            st.divider()
            model_type = st.selectbox(
                "추천 모델 선택",
                options=["hybrid", "knn", "svd", "ensemble", "content"],
                format_func=lambda x: {
                    "hybrid":   "⚡ 하이브리드 KNN (권장) — 기능매칭 + KNN + 올리브영 평점",
                    "knn":      "👥 KNN — 나와 피부가 비슷한 사람들이 좋아한 제품 추천",
                    "svd":      "🎬 SVD — 넷플릭스 방식 행렬 분해로 잠재 패턴 학습",
                    "ensemble": "🔀 앙상블 — KNN + BPR + LightGCN 3가지 모델 평균",
                    "content":  "🧴 기능매칭 — 피부 고민과 제품 기능만으로 매칭 (BERT 강화)",
                }[x],
                help="추천 알고리즘을 선택합니다. 권장 모델은 하이브리드 KNN입니다."
            )
            with st.expander("📌 모델별 설명 보기"):
                st.markdown("""
| 모델 | 어떻게 추천하는가 | 특징 |
|---|---|---|
| ⚡ **하이브리드 KNN (권장)** | 기능매칭 + 유사 사용자 경험 + 올리브영 평점 결합 | 피부 고민이 심할수록 기능매칭 비중 자동 증가 (동적 가중치) |
| 👥 **KNN** | 나와 Severity가 비슷한 사람 5명이 좋아한 제품 | Age/성별/기후/식단까지 반영한 유사 사용자 탐색 |
| 🎬 **SVD** | 15,000명 × 600개 평점 행렬을 분해하여 잠재 패턴 학습 | 넷플릭스가 사용하는 방식. 간접적 취향 패턴 반영 |
| 🔀 **앙상블** | KNN + BPR + LightGCN 세 모델 점수를 평균 | 단일 모델의 약점을 상호 보완 |
| 🧴 **기능매칭** | 피부 고민 → 필요 기능 → 제품 기능 일치도 계산 | BERT 한국어 모델로 상품명 의미까지 반영 |

> **동적 가중치란?** 피부 고민이 뚜렷할수록 기능매칭(α) 비중이 자동으로 높아지고, 고민이 약할수록 협업 필터링(β) 비중이 높아집니다.
                """)
            show_top3 = st.checkbox(
                "카테고리별 3개 추천 보기",
                value=False,
                help="체크하면 카테고리별 상위 3개 제품을 모두 표시합니다",
            )
            top_n = 3 if show_top3 else 1
            run_btn = st.button("✨ 추천 받기", type="primary",
                                use_container_width=True)

        with right:
            if run_btn:
                with st.spinner("추천 계산 중..."):
                    recs, _ = recommend(
                        user_scores, products, users, knn_mdl, scaler, inter,
                        model_type=model_type,
                        top_candidates=top_n,
                        svd_model=svd_mdl,
                        bpr_model=bpr_mdl,
                        lgcn_model=lgcn_mdl,
                        pid_to_emb=pid_to_emb,
                    )
                st.session_state["last_recs"] = recs
                st.session_state["last_scores"] = user_scores
                st.session_state["last_top_n"] = top_n

            if "last_recs" in st.session_state:
                recs = st.session_state["last_recs"]
                u_weights = get_user_weights(st.session_state["last_scores"])
                top_funcs = sorted(u_weights, key=u_weights.get, reverse=True)[:3]
                func_kor = " · ".join(FUNCTION_KOR.get(f, f) for f in top_funcs)
                scores = st.session_state["last_scores"]
                non_sens = {k: v for k, v in scores.items() if k in SEVERITY_COLS and k != "Sensitivity_Severity"}
                main_concern = max(non_sens, key=non_sens.get) if non_sens else SEVERITY_COLS[0]
                sens_str = " | 🌿 민감성 피부" if scores.get("Sensitivity_Severity", 0) > 0 else ""

                alpha_d, beta_d, gamma_d = get_dynamic_weights(scores)
                st.info(
                    f"주요 피부 고민: **{SEVERITY_LABELS[main_concern]}**{sens_str} | "
                    f"추천 기능 순위: {func_kor} | "
                    f"동적 가중치: α={alpha_d} / β={beta_d} / γ={gamma_d}"
                )
                per_cat = st.session_state.get("last_top_n", 1)
                st.subheader(f"✅ 맞춤 추천 {len(recs)}개 (카테고리별 {per_cat}개)")
                cat_recs: dict = {}
                cats_order: list = []
                for rec in recs:
                    cat = rec["category"]
                    if cat not in cat_recs:
                        cat_recs[cat] = []
                        cats_order.append(cat)
                    cat_recs[cat].append(rec)
                for cat in cats_order:
                    st.markdown(f"#### {CAT_DISPLAY.get(cat, cat)}")
                    items = cat_recs[cat]
                    cols = st.columns(3)
                    for i, rec in enumerate(items):
                        with cols[i % 3]:
                            render_card(rec)

                st.divider()
                render_baumann_card(get_baumann_type(st.session_state["last_scores"]))
            else:
                st.markdown("""
### 추천 방식 안내
| 구성 요소 | 가중치 | 설명 |
|-----------|--------|------|
| 기능 매칭 + BERT | **α (동적)** | 피부 문제 → 필요 기능 매핑 + 상품명 임베딩 |
| 협업 필터링 | **β (동적)** | KNN / BPR / LightGCN / SVD 선택 |
| 올리브영 평점 | **γ=0.2** | 올리브영 평균 별점 |

**피부 고민이 심할수록 α↑, 완만할수록 β↑ (자동 조정)**

**추천 카테고리:** 마스크팩 · 스킨케어 · 앰플/세럼 · 크림/로션 · 선케어
                """)

    # ══════════════════════════════════════
    # TAB 2 : 성능 비교
    # ══════════════════════════════════════
    with tab_eval:
        st.subheader("모델 성능 평가")

        st.markdown("### 🌿 추천 품질 평가 (올리브영 기준)")
        st.caption(
            "정답(ground truth) 없이도 측정 가능한 3가지 지표. "
            "추천받기 탭에서 먼저 추천을 받으면 자동으로 계산됩니다."
        )

        with st.expander("📌 지표 설명", expanded=True):
            st.markdown("""
| 지표 | 설명 | 의미 |
|---|---|---|
| **기능 적합도** | 추천 제품 기능이 사용자 피부 고민과 일치하는 비율 | 높을수록 피부 고민에 맞는 추천 |
| **올리브영 품질 점수** | 추천 제품 평균 평점 / DB 전체 평균 평점 | 1.0 이상이면 평균보다 좋은 제품 추천 |
| **카테고리 커버리지** | 5개 카테고리 중 추천된 카테고리 수 | 1.0이면 5단계 루틴 완성 |
            """)

        if "last_recs" in st.session_state and "last_scores" in st.session_state:
            recs_eval = st.session_state["last_recs"]
            user_scores_eval = st.session_state["last_scores"]

            user_weights_eval = get_user_weights(user_scores_eval)
            top_funcs_eval = set(
                sorted(user_weights_eval, key=user_weights_eval.get, reverse=True)[:3]
            )
            func_scores = []
            for rec in recs_eval:
                prod_funcs = set(
                    f.strip() for f in str(rec["functions"]).split("|") if f.strip()
                )
                overlap = len(top_funcs_eval & prod_funcs)
                func_scores.append(overlap / len(top_funcs_eval) if top_funcs_eval else 0)
            func_relevance = round(float(np.mean(func_scores)), 3) if func_scores else 0.0

            valid_ratings = [r["avg_rating"] for r in recs_eval if pd.notna(r["avg_rating"])]
            avg_rec_rating = float(np.mean(valid_ratings)) if valid_ratings else 0.0
            db_avg_rating = float(products["평균 평점"].mean())
            oy_quality = round(avg_rec_rating / db_avg_rating, 3) if db_avg_rating > 0 else 0.0

            covered_cats = len(set(r["category"] for r in recs_eval))
            cat_coverage = round(covered_cats / len(CATEGORIES), 3)

            q1, q2, q3 = st.columns(3)
            q1.metric("기능 적합도", f"{func_relevance:.3f}",
                      help="추천 제품 기능이 피부 고민과 일치하는 비율 (0~1, 높을수록 좋음)")
            q2.metric("올리브영 품질 점수", f"{oy_quality:.3f}",
                      f"DB 평균 대비 {'+' if oy_quality >= 1 else ''}{(oy_quality - 1) * 100:.1f}%",
                      help="추천 제품 평균 평점 / 전체 DB 평균 평점")
            q3.metric("카테고리 커버리지", f"{cat_coverage:.1%}",
                      f"{covered_cats}/5개 카테고리",
                      help="5단계 루틴이 얼마나 완성되었는지")

            with st.expander("세부 내역 보기"):
                detail_rows = []
                for rec in recs_eval:
                    prod_funcs = set(
                        f.strip() for f in str(rec["functions"]).split("|") if f.strip()
                    )
                    overlap = len(top_funcs_eval & prod_funcs)
                    nm = rec["name"]
                    detail_rows.append({
                        "카테고리": rec["category"],
                        "추천 제품": nm[:22] + "..." if len(nm) > 22 else nm,
                        "기능 일치": f"{overlap}/{len(top_funcs_eval)}",
                        "올리브영 평점": (
                            f"{rec['avg_rating']:.2f}" if pd.notna(rec["avg_rating"]) else "-"
                        ),
                        "최종 점수": f"{rec['final_score']:.3f}",
                    })
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)
        else:
            st.info("💡 먼저 '🎯 추천받기' 탭에서 추천을 받으면 품질 지표가 자동으로 계산됩니다.")

        st.divider()

        st.markdown("### 🔬 유사 사용자 탐색 정확도 (캐글 내부 검증)")
        st.caption(
            "캐글 15,000명 데이터 내부에서 유사 사용자를 얼마나 정확히 찾는지 검증합니다."
        )

        with st.expander("📌 평가 방법 및 모델 설명", expanded=False):
            st.markdown("""
**평가 방법: Leave-one-out**
- 각 사용자의 최고 평점 제품 1개를 숨긴 후 추천 생성
- 숨긴 제품이 추천 목록에 포함되는지 확인
- Precision@K: 상위 K개 중 정답 포함 비율
- NDCG@K: 정답의 순위가 높을수록 높은 점수

**비교 모델 5가지**
| 모델 | 설명 |
|---|---|
| 기능매칭 (BERT 강화) | 피부 고민 → 제품 기능 직접 매칭. BERT 상품명 임베딩 포함 |
| KNN Only | 유사 사용자 5명의 평점 기반 추천 |
| 하이브리드 KNN (권장) | 기능매칭 + KNN + 올리브영 평점 결합. 동적 가중치 |
| SVD (넷플릭스 방식) | 15,000명 × 600개 행렬 분해로 잠재 패턴 학습 |
| 앙상블 | KNN + BPR + LightGCN 평균으로 단일 모델 약점 보완 |

> **한계:** 이 평가는 Kaggle 데이터 내부 검증입니다. 올리브영 추천 품질은 섹션 A의 3가지 지표로 별도 평가합니다.
            """)

        col_n, col_btn = st.columns([2, 1])
        with col_n:
            n_eval = st.slider("평가 사용자 수", 50, 500, 200, 50,
                               help="많을수록 정확하지만 시간이 걸립니다")
        with col_btn:
            st.write(""); st.write("")
            eval_btn = st.button("▶ 평가 실행", type="primary")

        test_k = st.checkbox(
            "K값 최적화 실험 포함 (K=5,10,15,20 비교)",
            value=False,
            help="체크하면 KNN 이웃 수별 성능을 비교합니다. 시간이 더 걸립니다.",
        )

        if eval_btn:
            with st.spinner("평가 실행 중... (잠시 기다려 주세요)"):
                metrics_df = run_evaluation(
                    users, inter, products, knn_mdl, scaler,
                    n_eval=n_eval,
                    svd_model=svd_mdl,
                    bpr_model=bpr_mdl,
                    lgcn_model=lgcn_mdl,
                )
                st.session_state["metrics"] = metrics_df

                if test_k:
                    k_results = []
                    for k_val in [5, 10, 15, 20]:
                        feat_df = users[KNN_FEATURE_COLS].fillna(0)
                        sc_tmp = MinMaxScaler()
                        X_tmp = sc_tmp.fit_transform(feat_df)
                        knn_tmp = NearestNeighbors(
                            n_neighbors=k_val, metric="cosine", algorithm="brute"
                        )
                        knn_tmp.fit(X_tmp)
                        m = run_evaluation(users, inter, products, knn_tmp, sc_tmp,
                                           n_eval=n_eval, svd_model=svd_mdl,
                                           bpr_model=bpr_mdl, lgcn_model=lgcn_mdl)
                        hybrid_row = m[m["모델"] == "하이브리드 KNN (권장)"].iloc[0]
                        k_results.append({
                            "K값": k_val,
                            "Precision@15": hybrid_row.get("Precision@15", 0),
                            "NDCG@15": hybrid_row.get("NDCG@15", 0),
                        })
                    k_df = pd.DataFrame(k_results)
                    st.session_state["k_results"] = k_df
                    best_k = k_df.loc[k_df["NDCG@15"].idxmax(), "K값"]
                    st.success(f"✅ 최적 K값: {best_k} (NDCG@15 기준)")
                    st.dataframe(k_df, use_container_width=True)

        if "metrics" in st.session_state:
            mdf = st.session_state["metrics"]
            st.dataframe(mdf.set_index("모델"), use_container_width=True)

            import altair as alt

            model_domain = list(mdf["모델"].tolist())
            model_range  = MODEL_COLORS[:len(model_domain)]

            ndcg_cols = [c for c in mdf.columns if "NDCG" in c]
            melted = mdf[["모델"] + ndcg_cols].melt(
                id_vars="모델", value_vars=ndcg_cols,
                var_name="지표", value_name="NDCG",
            )
            chart = (
                alt.Chart(melted)
                .mark_bar()
                .encode(
                    x=alt.X("지표:N", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("NDCG:Q"),
                    color=alt.Color(
                        "모델:N",
                        scale=alt.Scale(domain=model_domain, range=model_range),
                    ),
                    xOffset="모델:N",
                    tooltip=["모델", "지표", "NDCG"],
                )
                .properties(height=320, title="NDCG@K 모델 비교")
            )
            st.altair_chart(chart, use_container_width=True)

            prec_cols = [c for c in mdf.columns if "Precision" in c]
            melted_p = mdf[["모델"] + prec_cols].melt(
                id_vars="모델", value_vars=prec_cols,
                var_name="지표", value_name="Precision",
            )
            chart_p = (
                alt.Chart(melted_p)
                .mark_bar()
                .encode(
                    x=alt.X("지표:N", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Precision:Q"),
                    color=alt.Color(
                        "모델:N",
                        scale=alt.Scale(domain=model_domain, range=model_range),
                    ),
                    xOffset="모델:N",
                    tooltip=["모델", "지표", "Precision"],
                )
                .properties(height=320, title="Precision@K 모델 비교")
            )
            st.altair_chart(chart_p, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 3 : EDA / 데이터 분석
    # ══════════════════════════════════════
    with tab_data:
        st.subheader("EDA / 데이터 분석")
        import altair as alt

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kaggle 사용자", f"{len(users):,}명")
        c2.metric("Interactions",  f"{len(inter):,}개")
        c3.metric("학습 레이블",   f"{len(y_labels):,}명")
        c4.metric("올리브영 제품", f"{len(products):,}개")

        st.divider()

        st.markdown("### 🛒 올리브영 제품 DB 분석")

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**카테고리별 제품 수**")
            cat_cnt = products.groupby("카테고리").size().reset_index()
            cat_cnt.columns = ["카테고리", "제품 수"]
            bar1 = (
                alt.Chart(cat_cnt)
                .mark_bar()
                .encode(
                    x=alt.X("카테고리:N", sort="-y", axis=alt.Axis(labelAngle=-20)),
                    y="제품 수:Q",
                    color=alt.Color("카테고리:N", legend=None,
                                    scale=alt.Scale(domain=CATEGORIES,
                                                    range=list(CAT_COLOR.values()))),
                    tooltip=["카테고리", "제품 수"],
                )
                .properties(height=260)
            )
            st.altair_chart(bar1, use_container_width=True)

        with col_b:
            st.write("**기능별 제품 수**")
            all_funcs = []
            for fs in products["기능(Function)"].dropna():
                all_funcs += [f.strip() for f in str(fs).split("|") if f.strip()]
            fc = pd.Series(all_funcs).value_counts().reset_index()
            fc.columns = ["기능", "제품 수"]
            fc["기능_KOR"] = fc["기능"].map(FUNCTION_KOR).fillna(fc["기능"])
            bar2 = (
                alt.Chart(fc)
                .mark_bar(color="#45B7D1")
                .encode(
                    x="제품 수:Q",
                    y=alt.Y("기능_KOR:N", sort="-x"),
                    tooltip=["기능_KOR", "기능", "제품 수"],
                )
                .properties(height=260)
            )
            st.altair_chart(bar2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.write("**올리브영 평균 평점 분포**")
            rating_data = products["평균 평점"].dropna().reset_index(drop=True).to_frame()
            rating_data.columns = ["평균 평점"]
            hist_rating = (
                alt.Chart(rating_data)
                .mark_bar(color="#FF6B6B", opacity=0.85)
                .encode(
                    x=alt.X("평균 평점:Q", bin=alt.Bin(step=0.1), title="평균 평점"),
                    y=alt.Y("count()", title="제품 수"),
                    tooltip=[alt.Tooltip("평균 평점:Q", bin=alt.Bin(step=0.1)), "count()"],
                )
                .properties(height=260)
            )
            st.altair_chart(hist_rating, use_container_width=True)

        with col_d:
            st.write("**제품 가격대 분포 (상위 3% 제외)**")
            price_data = products["할인가(원)"].dropna()
            price_data = price_data[price_data <= price_data.quantile(0.97)].reset_index(drop=True).to_frame()
            price_data.columns = ["할인가"]
            hist_price = (
                alt.Chart(price_data)
                .mark_bar(color="#96CEB4", opacity=0.85)
                .encode(
                    x=alt.X("할인가:Q", bin=alt.Bin(maxbins=30), title="할인가(원)"),
                    y=alt.Y("count()", title="제품 수"),
                    tooltip=[alt.Tooltip("할인가:Q", bin=alt.Bin(maxbins=30)), "count()"],
                )
                .properties(height=260)
            )
            st.altair_chart(hist_price, use_container_width=True)

        st.divider()

        st.markdown("### 👤 Kaggle 사용자 피부 분석 (15,000명)")

        col_e, col_f = st.columns(2)
        with col_e:
            st.write("**Severity 항목별 분포 (0 초과 사용자만)**")
            sev_long = users[SEVERITY_COLS].copy()
            sev_long.columns = [SEVERITY_LABELS[c] for c in SEVERITY_COLS]
            sev_long = sev_long.melt(var_name="피부 고민", value_name="Severity")
            sev_long = sev_long[sev_long["Severity"] > 0]
            hist_sev = (
                alt.Chart(sev_long)
                .mark_bar(opacity=0.75)
                .encode(
                    x=alt.X("Severity:Q", bin=alt.Bin(maxbins=20), title="Severity 점수"),
                    y=alt.Y("count()", title="사용자 수"),
                    color=alt.Color("피부 고민:N"),
                    tooltip=["피부 고민",
                             alt.Tooltip("Severity:Q", bin=alt.Bin(maxbins=20)),
                             "count()"],
                )
                .properties(height=280)
            )
            st.altair_chart(hist_sev, use_container_width=True)

        with col_f:
            st.write("**Severity 간 상관관계 히트맵**")
            corr = users[SEVERITY_COLS].corr().round(3)
            corr_long = corr.reset_index().melt(id_vars="index")
            corr_long.columns = ["x", "y", "상관계수"]
            corr_long["x_kor"] = corr_long["x"].map(SEVERITY_LABELS)
            corr_long["y_kor"] = corr_long["y"].map(SEVERITY_LABELS)
            heatmap = (
                alt.Chart(corr_long)
                .mark_rect()
                .encode(
                    x=alt.X("x_kor:N", title=None, axis=alt.Axis(labelAngle=-30)),
                    y=alt.Y("y_kor:N", title=None),
                    color=alt.Color(
                        "상관계수:Q",
                        scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                    ),
                    tooltip=["x_kor", "y_kor",
                             alt.Tooltip("상관계수:Q", format=".3f")],
                )
                .properties(height=280)
            )
            text_layer = heatmap.mark_text(fontSize=11).encode(
                text=alt.Text("상관계수:Q", format=".2f"),
                color=alt.condition(
                    "abs(datum['상관계수']) > 0.5",
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
            st.altair_chart(heatmap + text_layer, use_container_width=True)

        col_g, col_h = st.columns(2)
        with col_g:
            st.write("**민감성 피부 비율**")
            sens_cnt = users["Sensitivity_Severity"].apply(
                lambda v: "민감성" if v > 0 else "비민감성"
            ).value_counts().reset_index()
            sens_cnt.columns = ["구분", "사용자 수"]
            pie = (
                alt.Chart(sens_cnt)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("사용자 수:Q"),
                    color=alt.Color(
                        "구분:N",
                        scale=alt.Scale(
                            domain=["민감성", "비민감성"],
                            range=["#FF6B6B", "#4ECDC4"],
                        ),
                    ),
                    tooltip=["구분", "사용자 수"],
                )
                .properties(height=260)
            )
            st.altair_chart(pie, use_container_width=True)

        with col_h:
            st.write("**Severity 통계 요약**")
            sev_stats = users[SEVERITY_COLS].describe().T.reset_index()
            sev_stats.columns = ["컬럼"] + list(sev_stats.columns[1:])
            sev_stats["피부 고민"] = sev_stats["컬럼"].map(SEVERITY_LABELS)
            st.dataframe(
                sev_stats[["피부 고민", "mean", "std", "min", "50%", "max"]]
                .rename(columns={"mean": "평균", "std": "표준편차",
                                 "min": "최솟값", "50%": "중앙값", "max": "최댓값"})
                .set_index("피부 고민"),
                use_container_width=True,
            )

        st.divider()

        st.markdown("### 🔗 Kaggle 사용자-제품 상호작용")

        col_i, col_j = st.columns(2)
        with col_i:
            st.write("**사용자 평점 분포**")
            rating_dist = inter["User_Rating"].dropna().reset_index(drop=True).to_frame()
            rating_dist.columns = ["평점"]
            hist_ur = (
                alt.Chart(rating_dist)
                .mark_bar(color="#FFA94D", opacity=0.85)
                .encode(
                    x=alt.X("평점:Q", bin=alt.Bin(step=0.5), title="User Rating"),
                    y=alt.Y("count()", title="건수"),
                    tooltip=[alt.Tooltip("평점:Q", bin=alt.Bin(step=0.5)), "count()"],
                )
                .properties(height=260)
            )
            st.altair_chart(hist_ur, use_container_width=True)

        with col_j:
            st.write("**제품별 interaction 수 분포**")
            prod_cnt_s = inter.groupby("Product_ID").size().reset_index()
            prod_cnt_s.columns = ["Product_ID", "interaction 수"]
            hist_pc = (
                alt.Chart(prod_cnt_s)
                .mark_bar(color="#45B7D1", opacity=0.85)
                .encode(
                    x=alt.X("interaction 수:Q", bin=alt.Bin(maxbins=20),
                            title="제품당 interaction 수"),
                    y=alt.Y("count()", title="제품 수"),
                    tooltip=[alt.Tooltip("interaction 수:Q", bin=alt.Bin(maxbins=20)),
                             "count()"],
                )
                .properties(height=260)
            )
            st.altair_chart(hist_pc, use_container_width=True)
            st.caption("※ 모든 사용자(15,000명)가 카테고리별 1개씩 정확히 5개 평점을 남긴 구조")

        st.divider()

        st.markdown("### 🗺️ Severity → 기능 매핑 룰")
        map_rows = []
        for sev, funcs in SEVERITY_FUNCTION_MAP.items():
            map_rows.append({
                "피부 문제": f"{SEVERITY_EMOJI[sev]} {SEVERITY_LABELS[sev]}",
                "매핑 기능 (가중치 순)": " → ".join(
                    FUNCTION_KOR.get(f, f) for f in funcs
                ),
            })
        st.table(pd.DataFrame(map_rows))


if __name__ == "__main__":
    main()
