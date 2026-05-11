"""
스킨케어 추천 시스템 Phase 2
하이브리드 추천 엔진: 기능매칭(50%) + KNN평점(30%) + 올리브영평점(20%)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import glob
import io
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
    page_title="스킨케어 추천 시스템",
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

# 실제 DB 기능: Moisturizing | SkinBarrier | Firming | Whitning | Exfoliation
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

    # 올리브영 평점 0~1 정규화
    max_oy = products["평균 평점"].max() or 1
    products["oy_norm"] = products["평균 평점"].fillna(0) / max_oy

    return users, inter, y_labels, products


# ─────────────────────────────────────────────
# 2. KNN 모델 구축 (Kaggle 사용자 피부 프로파일)
# ─────────────────────────────────────────────
@st.cache_resource
def build_knn(_users_df):
    feats = _users_df[SEVERITY_COLS].fillna(0).values
    scaler = MinMaxScaler()
    feats_scaled = scaler.fit_transform(feats)
    knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
    knn.fit(feats_scaled)
    return knn, scaler


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
    """ResNet50 회귀 + MobileNetV3-Large 분류 모델 로드 (CPU)"""
    # 회귀 모델: fc = Sequential(Linear(2048,256), ReLU, Dropout, Linear(256,5))
    # 출력 5개 중 앞 4개 사용: Acne / Dryness / Aging / Pigmentation
    reg = tvmodels.resnet50()
    reg.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 5),
    )
    reg.load_state_dict(
        torch.load(REGRESSOR_PATH, map_location="cpu", weights_only=False)
    )
    reg.eval()

    # 분류 모델: MobileNetV3-Large, classifier[3] 교체
    # classifier[3] = Sequential(Linear(1280,128), ReLU, Dropout, Linear(128,2))
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
    """업로드 이미지 → 피부 severity 5개 수치 반환"""
    reg, cls = load_skin_models()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _IMG_TRANSFORM(img).unsqueeze(0)          # (1, 3, 224, 224)

    with torch.no_grad():
        reg_out = reg(x).squeeze().tolist()        # [Acne, Dryness, Aging, Pigmentation]
        cls_pred = int(cls(x).argmax(dim=1).item()) # 0 or 1

    return {
        "Acne_Severity":         float(np.clip(reg_out[0], 0.0, 10.0)),
        "Dryness_Severity":      float(np.clip(reg_out[1], 0.0, 10.0)),
        "Aging_Severity":        float(np.clip(reg_out[2], 0.0,  4.2)),
        "Pigmentation_Severity": float(np.clip(reg_out[3], 0.0,  6.0)),
        "Sensitivity_Severity":  6.49 if cls_pred == 1 else 0.0,
    }


# ─────────────────────────────────────────────
# 3. Severity → 기능 가중치
# ─────────────────────────────────────────────
def get_user_weights(user_scores: dict) -> dict:
    """피부 severity 점수 → 기능별 가중치 (합계 1로 정규화)"""
    weights: dict = {}
    for col, funcs in SEVERITY_FUNCTION_MAP.items():
        w = max(0.0, float(user_scores.get(col, 0))) / 10.0
        for f in funcs:
            weights[f] = weights.get(f, 0) + w
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def feature_match_score(func_str, user_weights: dict) -> float:
    if pd.isna(func_str):
        return 0.0
    return sum(user_weights.get(f.strip(), 0) for f in str(func_str).split("|"))


# ─────────────────────────────────────────────
# 4. KNN 평점 예측
# ─────────────────────────────────────────────
def knn_product_scores(user_scores: dict, users_df, knn_mdl, scaler, inter_df) -> dict:
    feat = np.array([[float(user_scores.get(c, 0)) for c in SEVERITY_COLS]])
    feat_sc = scaler.transform(feat)
    _, idx = knn_mdl.kneighbors(feat_sc)
    neighbor_ids = users_df.iloc[idx[0]]["User_ID"].tolist()
    nb_inter = inter_df[inter_df["User_ID"].isin(neighbor_ids)]
    return nb_inter.groupby("Product_ID")["User_Rating"].mean().to_dict()


# ─────────────────────────────────────────────
# 5. 하이브리드 추천 엔진
# ─────────────────────────────────────────────
def recommend(
    user_scores: dict,
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    knn_mdl,
    scaler,
    inter_df: pd.DataFrame,
    alpha: float = 0.5,   # 기능매칭
    beta: float = 0.3,    # KNN평점
    gamma: float = 0.2,   # 올리브영평점
    model_type: str = "hybrid",
    top_candidates: int = 3,
):
    """카테고리별 상위 top_candidates 후보 중 최고 점수 1개 반환, 총 5개"""
    user_weights = get_user_weights(user_scores)
    df = products_df.copy()

    # 기능 매칭 점수 (0~1)
    df["feat_raw"] = df["기능(Function)"].apply(
        lambda x: feature_match_score(x, user_weights))
    max_feat = df["feat_raw"].max() or 1.0
    df["feat_score"] = df["feat_raw"] / max_feat

    # KNN 점수 (0~1)
    if model_type in ("knn", "hybrid"):
        knn_sc = knn_product_scores(user_scores, users_df, knn_mdl, scaler, inter_df)
        df["knn_score"] = df["Product_ID"].map(knn_sc).fillna(0) / 5.0
    else:
        df["knn_score"] = 0.0

    # 최종 점수
    if model_type == "content":
        df["final_score"] = df["feat_score"]
    elif model_type == "knn":
        df["final_score"] = df["knn_score"]
    else:  # hybrid
        df["final_score"] = (
            alpha * df["feat_score"] +
            beta  * df["knn_score"] +
            gamma * df["oy_norm"]
        )

    # 카테고리별 top_candidates개 반환
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
                "knn_score":   round(float(best["knn_score"]), 4),
                "oy_score":    round(float(best["oy_norm"]), 4),
                "final_score": round(float(best["final_score"]), 4),
                "price":       best["할인가(원)"],
                "oy_rank":     best["올리브영 순위"],
                "avg_rating":  best["평균 평점"],
            })
    return recs, df


# ─────────────────────────────────────────────
# 6. Leave-one-out 평가
# ─────────────────────────────────────────────
def run_evaluation(users_df, inter_df, products_df, knn_mdl, scaler,
                   n_eval: int = 200, k_list=(1, 3, 5)):
    """Content-based / KNN only / Hybrid 3모델 비교"""
    valid_pids = set(products_df["Product_ID"].tolist())
    inter_v = inter_df[inter_df["Product_ID"].isin(valid_pids)].copy()

    # 유효 interaction ≥ 2인 사용자 샘플링
    cnt = inter_v.groupby("User_ID").size()
    eligible = cnt[cnt >= 2].index.tolist()
    np.random.seed(42)
    sample = np.random.choice(eligible, min(n_eval, len(eligible)), replace=False)

    models = ["content", "knn", "hybrid"]
    hits = {m: {k: 0   for k in k_list} for m in models}
    ndcg = {m: {k: 0.0 for k in k_list} for m in models}
    total = {m: 0 for m in models}
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

        for model in models:
            recs, score_df = recommend(
                u_scores, products_df, users_df, knn_mdl, scaler, inter_df,
                model_type=model
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

    MODEL_LABELS = {"content": "Content-based", "knn": "KNN Only", "hybrid": "하이브리드"}
    rows = []
    for m in models:
        t = total[m] or 1
        row = {"모델": MODEL_LABELS[m]}
        for k in k_list:
            row[f"Precision@{k}"] = round(hits[m][k] / t, 4)
            row[f"NDCG@{k}"]      = round(ndcg[m][k] / t, 4)
        row["Category Coverage"] = round(len(cat_cov[m]) / len(CATEGORIES), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 7. UI 컴포넌트
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

    # score bars
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
    <span style="font-size:22px;font-weight:bold;color:{color}">
      {rec['final_score']:.3f}
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
      <span class="score-label">기능매칭 (α=0.5)</span>
      <div style="font-weight:bold">{rec['feat_score']:.3f}</div>
      {bar(rec['feat_score'])}
    </div>
    <div>
      <span class="score-label">KNN평점 (β=0.3)</span>
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


# ─────────────────────────────────────────────
# 8. MAIN APP
# ─────────────────────────────────────────────
def main():
    st.title("🧴 COSDUO — 얼굴 이미지 기반 스킨케어 루틴 추천")
    st.caption(
        "ResNet50 피부 분석 → 하이브리드 추천 (기능매칭 50% + KNN 30% + 올리브영 평점 20%) | "
        "이화여자대학교 데이터사이언스대학원 DUO COS"
    )

    with st.spinner("데이터 로딩 중..."):
        users, inter, y_labels, products = load_data()
        knn_mdl, scaler = build_knn(users)

    tab_rec, tab_eval, tab_data = st.tabs(
        ["🎯 추천받기", "📊 성능 비교", "📋 데이터 현황"]
    )

    # ══════════════════════════════════════
    # TAB 1 : 추천받기
    # ══════════════════════════════════════
    with tab_rec:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.subheader("피부 상태 입력")
            input_mode = st.radio(
                "입력 방법",
                ["슬라이더 직접 입력", "데이터셋에서 선택", "📷 얼굴 사진으로 분석"],
                horizontal=True,
            )

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
                            help="0: 없음 · 5: 보통 · 10: 심각",
                        )

            elif input_mode == "데이터셋에서 선택":
                valid_y = y_labels.dropna(subset=SEVERITY_COLS)
                sample_y = valid_y.sample(min(200, len(valid_y)), random_state=42)

                def _label(row):
                    age = f"{row['age']:.0f}세" if pd.notna(row.get("age")) else "?세"
                    return (
                        f"ID {row['image_id']} | {age} | "
                        f"여드름{row['Acne_Severity']:.1f} "
                        f"건조{row['Dryness_Severity']:.1f} "
                        f"노화{row['Aging_Severity']:.1f}"
                    )

                opts = [_label(r) for _, r in sample_y.iterrows()]
                sel_idx = st.selectbox("사용자 선택", range(len(opts)),
                                       format_func=lambda i: opts[i])
                sel_row = sample_y.iloc[sel_idx]
                user_scores = {c: float(sel_row.get(c, 0) or 0)
                               for c in SEVERITY_COLS}

                st.divider()
                for col in SEVERITY_COLS:
                    emoji = SEVERITY_EMOJI[col]
                    label = SEVERITY_LABELS[col]
                    val   = user_scores[col]
                    bar_w = min(10, int(val))
                    st.markdown(
                        f"**{emoji} {label}** `{val:.1f}` "
                        f"{'█' * bar_w}{'░' * (10 - bar_w)}"
                    )

            else:  # 📷 얼굴 사진으로 분석
                uploaded = st.file_uploader(
                    "얼굴 사진 업로드 (jpg / png)",
                    type=["jpg", "jpeg", "png"],
                )

                if uploaded is not None:
                    st.image(uploaded, caption="업로드된 사진", use_container_width=True)
                    with st.spinner("AI 피부 분석 중..."):
                        image_bytes = uploaded.read()
                        user_scores = infer_skin_scores(image_bytes)

                    st.success("분석 완료!")
                    st.divider()
                    for col in SEVERITY_COLS:
                        emoji = SEVERITY_EMOJI[col]
                        label = SEVERITY_LABELS[col]
                        val   = user_scores[col]
                        bar_w = min(10, int(val * 10 / 10))
                        st.markdown(
                            f"**{emoji} {label}** `{val:.2f}` "
                            f"{'█' * bar_w}{'░' * (10 - bar_w)}"
                        )
                else:
                    st.info("jpg 또는 png 파일을 업로드하면 AI가 피부를 자동 분석합니다.")
                    # 사진이 없으면 추천 버튼 비활성화를 위해 기본값 설정
                    user_scores = {c: 0.0 for c in SEVERITY_COLS}

            st.divider()
            model_type = st.selectbox(
                "모델",
                ["hybrid", "content", "knn"],
                format_func=lambda x: {
                    "hybrid":  "⚡ 하이브리드 (권장)",
                    "content": "🔍 Content-based (기능 매칭만)",
                    "knn":     "👥 KNN (협업 필터링만)",
                }[x],
            )
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
                non_sens = {k: v for k, v in scores.items() if k != "Sensitivity_Severity"}
                main_concern = max(non_sens, key=non_sens.get)
                sens_str = " | 🌿 민감성 피부" if scores.get("Sensitivity_Severity", 0) > 0 else ""
                st.info(
                    f"주요 피부 고민: **{SEVERITY_LABELS[main_concern]}**{sens_str} | "
                    f"추천 기능 순위: {func_kor}"
                )
                per_cat = st.session_state.get("last_top_n", 1)
                st.subheader(f"✅ 맞춤 추천 {len(recs)}개 (카테고리별 {per_cat}개)")
                for rec in recs:
                    render_card(rec)
            else:
                st.markdown("""
### 추천 방식 안내
| 구성 요소 | 가중치 | 설명 |
|-----------|--------|------|
| 기능 매칭 | **50%** | 피부 문제 → 필요 기능 매핑 |
| KNN 평점  | **30%** | 유사 사용자(K=10) 평균 평점 |
| 올리브영 평점 | **20%** | 올리브영 평균 별점 |

**추천 카테고리:** 마스크팩 · 스킨케어 · 앰플/세럼 · 크림/로션 · 선케어
                """)

    # ══════════════════════════════════════
    # TAB 2 : 성능 비교
    # ══════════════════════════════════════
    with tab_eval:
        st.subheader("모델 성능 평가")

        # ── 섹션 A: 추천 품질 평가 (올리브영 기준) ──────────────────
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

> **교수님 질문에 대한 답변:** 이 시스템은 정답이 정해진 문제가 아닙니다.
> 올리브영 추천 품질은 위 3가지 지표로 평가하고,
> 유사 사용자 탐색 정확도는 아래 섹션 B에서 캐글 데이터로 별도 검증합니다.
            """)

        if "last_recs" in st.session_state and "last_scores" in st.session_state:
            recs_eval = st.session_state["last_recs"]
            user_scores_eval = st.session_state["last_scores"]

            # 지표 1: 기능 적합도
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

            # 지표 2: 올리브영 품질 점수
            valid_ratings = [
                r["avg_rating"] for r in recs_eval if pd.notna(r["avg_rating"])
            ]
            avg_rec_rating = float(np.mean(valid_ratings)) if valid_ratings else 0.0
            db_avg_rating = float(products["평균 평점"].mean())
            oy_quality = round(avg_rec_rating / db_avg_rating, 3) if db_avg_rating > 0 else 0.0

            # 지표 3: 카테고리 커버리지
            covered_cats = len(set(r["category"] for r in recs_eval))
            cat_coverage = round(covered_cats / len(CATEGORIES), 3)

            q1, q2, q3 = st.columns(3)
            q1.metric(
                "기능 적합도",
                f"{func_relevance:.3f}",
                help="추천 제품 기능이 피부 고민과 일치하는 비율 (0~1, 높을수록 좋음)",
            )
            q2.metric(
                "올리브영 품질 점수",
                f"{oy_quality:.3f}",
                f"DB 평균 대비 {'+' if oy_quality >= 1 else ''}{(oy_quality - 1) * 100:.1f}%",
                help="추천 제품 평균 평점 / 전체 DB 평균 평점 (1.0 이상이면 평균 이상 품질)",
            )
            q3.metric(
                "카테고리 커버리지",
                f"{cat_coverage:.1%}",
                f"{covered_cats}/5개 카테고리",
                help="5단계 루틴이 얼마나 완성되었는지 (1.0 = 완전한 루틴)",
            )

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
            st.info(
                "💡 먼저 '🎯 추천받기' 탭에서 추천을 받으면 품질 지표가 자동으로 계산됩니다."
            )

        st.divider()

        # ── 섹션 B: 유사 사용자 탐색 정확도 (캐글 내부 검증) ──────────
        st.markdown("### 🔬 유사 사용자 탐색 정확도 (캐글 내부 검증)")
        st.caption(
            "캐글 15,000명 데이터 내부에서 유사 사용자를 얼마나 정확히 찾는지 검증합니다. "
            "올리브영 추천 품질 평가(섹션 A)와는 별개입니다."
        )

        with st.expander("📌 평가 방법 안내", expanded=False):
            st.markdown("""
- **Leave-one-out**: 각 Kaggle 사용자의 최고 평점 제품을 hold-out 후 추천
- **유효 제품**: 올리브영 DB(601개)에 매칭된 Product_ID만 대상
- **Precision@K**: 상위 K개 추천 중 hold-out 제품이 포함된 비율
- **NDCG@K**: 순위 가중 평가 지표 (`1/log₂(rank+1)`)
- **Category Coverage**: 추천된 고유 카테고리 수 / 전체 카테고리(5개)
- **한계**: 캐글 데이터 기반 검증이므로 올리브영 추천 품질과 직접 연결되지 않음
            """)

        col_n, col_btn = st.columns([2, 1])
        with col_n:
            n_eval = st.slider(
                "평가 사용자 수", 50, 500, 200, 50,
                help="많을수록 정확하지만 시간이 걸립니다",
            )
        with col_btn:
            st.write(""); st.write("")
            eval_btn = st.button("▶ 평가 실행", type="primary")

        if eval_btn:
            with st.spinner("평가 실행 중... (잠시 기다려 주세요)"):
                metrics_df = run_evaluation(
                    users, inter, products, knn_mdl, scaler, n_eval=n_eval
                )
            st.session_state["metrics"] = metrics_df

        if "metrics" in st.session_state:
            mdf = st.session_state["metrics"]
            st.dataframe(
                mdf.set_index("모델").style.highlight_max(axis=0, color="#d4f4dd"),
                use_container_width=True,
            )

            import altair as alt
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
                        scale=alt.Scale(
                            domain=["Content-based", "KNN Only", "하이브리드"],
                            range=["#4ECDC4", "#45B7D1", "#FF6B6B"],
                        ),
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
                        scale=alt.Scale(
                            domain=["Content-based", "KNN Only", "하이브리드"],
                            range=["#4ECDC4", "#45B7D1", "#FF6B6B"],
                        ),
                    ),
                    xOffset="모델:N",
                    tooltip=["모델", "지표", "Precision"],
                )
                .properties(height=320, title="Precision@K 모델 비교")
            )
            st.altair_chart(chart_p, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 3 : 데이터 현황
    # ══════════════════════════════════════
    with tab_data:
        st.subheader("데이터셋 현황")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kaggle 사용자", f"{len(users):,}명")
        c2.metric("Interactions",  f"{len(inter):,}개")
        c3.metric("우리 사용자",   f"{len(y_labels):,}명")
        c4.metric("올리브영 제품", f"{len(products):,}개")

        st.divider()
        import altair as alt

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
                                    scale=alt.Scale(
                                        domain=CATEGORIES,
                                        range=list(CAT_COLOR.values()))),
                    tooltip=["카테고리", "제품 수"],
                )
                .properties(height=280)
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
                .properties(height=280)
            )
            st.altair_chart(bar2, use_container_width=True)

        st.divider()
        st.write("**Severity → 기능 매핑 룰**")
        map_rows = []
        for sev, funcs in SEVERITY_FUNCTION_MAP.items():
            map_rows.append({
                "피부 문제": f"{SEVERITY_EMOJI[sev]} {SEVERITY_LABELS[sev]}",
                "매핑 기능 (가중치 순)": " → ".join(
                    FUNCTION_KOR.get(f, f) for f in funcs
                ),
            })
        st.table(pd.DataFrame(map_rows))

        st.divider()
        st.write("**사용자 피부 점수 분포 (Kaggle 15,000명)**")
        sev_stats = users[SEVERITY_COLS].describe().T.reset_index()
        sev_stats.columns = ["컬럼"] + list(sev_stats.columns[1:])
        sev_stats["피부 문제"] = sev_stats["컬럼"].map(SEVERITY_LABELS)
        st.dataframe(
            sev_stats[["피부 문제", "mean", "std", "min", "50%", "max"]]
            .rename(columns={"mean": "평균", "std": "표준편차",
                             "min": "최솟값", "50%": "중앙값", "max": "최댓값"})
            .set_index("피부 문제"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
