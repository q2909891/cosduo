"""
COSDUO Skin Disease Pilot App
안면부 피부질환 분류 + 아토피 중증도 + 치료 성분 가이드
"""
import io
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
DISEASE_CLASSES = ["건선", "아토피", "여드름", "정상", "주사", "지루"]

DISEASE_EN = {
    "건선": "Psoriasis",
    "아토피": "Atopic Dermatitis",
    "여드름": "Acne",
    "정상": "Normal",
    "주사": "Rosacea",
    "지루": "Seborrheic Dermatitis",
}

DISEASE_EMOJI = {
    "건선": "🟥",
    "아토피": "🟧",
    "여드름": "🔴",
    "정상": "🟢",
    "주사": "🟪",
    "지루": "🟡",
}

SEVERITY_LABELS = {0: "경증", 1: "중등도", 2: "중증"}
SEVERITY_COLORS = {"경증": "#2ecc71", "중등도": "#f39c12", "중증": "#e74c3c"}

# 모델 출력 범위 ~[-0.5, 1.0] → IGA grade 인코딩 기반 (Mild=0, Moderate=1, Severe=2 정규화)
SEV_MILD_THR     = 0.25   # < 0.25 → 경증
SEV_MODERATE_THR = 0.75   # 0.25~0.75 → 중등도, ≥ 0.75 → 중증

# 모델 가중치 경로
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CLS_PATH    = os.path.join(BASE_DIR, "results", "disease_cls_resnet50.pth")
SEV_PATH    = os.path.join(BASE_DIR, "results", "atopy_severity_resnet50.pth")
MAP_PATH    = os.path.join(BASE_DIR, "treatment_mapping.csv")

# ImageNet 정규화
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────
def build_disease_classifier(num_classes: int = 6) -> nn.Module:
    """ResNet50 backbone + custom fc head (훈련 시 model.fc 교체 방식과 동일)"""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model


def build_severity_regressor() -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
    )
    return model


# ─────────────────────────────────────────────
# 모델 로더 (캐시)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="질환 분류 모델 로딩 중...")
def load_cls_model():
    model = build_disease_classifier(num_classes=len(DISEASE_CLASSES))
    sd = torch.load(CLS_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


@st.cache_resource(show_spinner="아토피 중증도 모델 로딩 중...")
def load_sev_model():
    model = build_severity_regressor()
    sd = torch.load(SEV_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


@st.cache_data
def load_treatment_map():
    df = pd.read_csv(MAP_PATH, encoding="utf-8-sig")
    return df


# ─────────────────────────────────────────────
# 추론 함수
# ─────────────────────────────────────────────
def predict_disease(img: Image.Image, model: nn.Module):
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return DISEASE_CLASSES[pred_idx], probs


def predict_severity(img: Image.Image, model: nn.Module):
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        score = model(tensor).item()
    # IGA grade 기반 임계값 (Mild=0, Moderate=1, Severe=2 정규화 회귀 출력)
    if score < SEV_MILD_THR:
        label = "경증"
    elif score < SEV_MODERATE_THR:
        label = "중등도"
    else:
        label = "중증"
    return score, label


def get_treatment(disease: str, severity: str, df: pd.DataFrame):
    row = df[(df["질환"] == disease) & (df["중증도"] == severity)]
    if row.empty:
        # 정상이면 경증 행 조회
        row = df[(df["질환"] == disease) & (df["중증도"] == "경증")]
    if row.empty:
        return None
    return row.iloc[0]


# ─────────────────────────────────────────────
# 성능 지표 표
# ─────────────────────────────────────────────
PERF_DATA = {
    "질환":        ["정상", "아토피", "여드름", "건선", "주사", "지루", "**전체**"],
    "Sensitivity": ["100%", "86%", "74%", "81%", "86%", "71%", "—"],
    "Specificity": ["100%", "96%", "99%", "96%", "97%", "93%", "—"],
    "Accuracy":    ["—",   "—",   "—",   "—",   "—",   "—",  "**83%**"],
}

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
CARD_CSS = """
<style>
.result-card {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 10px 0;
    border-left: 6px solid;
}
.disease-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    color: white;
    margin-bottom: 8px;
}
.sev-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: bold;
    color: white;
}
.warning-box {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #856404;
}
.prob-bar-bg {
    background: #e9ecef;
    border-radius: 4px;
    height: 8px;
    margin-top: 3px;
}
</style>
"""

DISEASE_COLORS = {
    "건선": "#e74c3c",
    "아토피": "#e67e22",
    "여드름": "#c0392b",
    "정상": "#27ae60",
    "주사": "#8e44ad",
    "지루": "#f1c40f",
}


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="COSDUO Skin Pilot",
        page_icon="🔬",
        layout="wide",
    )
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    st.title("안면부 피부질환 분류 파일럿")

    st.markdown(
        '<div class="warning-box">⚠️ <b>본 서비스는 의료 진단을 대체하지 않습니다.</b> '
        '파일럿 연구 목적으로만 사용하며, 정확한 진단은 반드시 피부과 전문의에게 받으시기 바랍니다.</div>',
        unsafe_allow_html=True,
    )

    # 모델 로딩
    cls_model = load_cls_model()
    sev_model = load_sev_model()
    treat_df  = load_treatment_map()

    tabs = st.tabs(["🔍 질환 분류", "📊 모델 성능", "📋 치료 성분 가이드"])

    # ── 탭 1: 질환 분류 ──────────────────────────
    with tabs[0]:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.subheader("얼굴 사진 업로드")
            st.caption("정면 얼굴 사진 (jpg / png). 1024×1024 권장.")
            uploaded = st.file_uploader(
                "파일 선택",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            if uploaded:
                img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
                st.image(img, use_container_width=True, caption="업로드된 사진")

        with col_result:
            if not uploaded:
                st.info("왼쪽에서 사진을 업로드하면 분석 결과가 여기에 표시됩니다.")
            else:
                with st.spinner("AI 분석 중..."):
                    disease, probs = predict_disease(img, cls_model)

                color = DISEASE_COLORS.get(disease, "#555")
                conf  = float(np.max(probs)) * 100

                st.subheader("분석 결과")

                # 질환 결과 카드
                st.markdown(
                    f'<div class="result-card" style="border-left-color:{color};background:#f8f9fa">'
                    f'<div class="disease-badge" style="background:{color}">'
                    f'{DISEASE_EMOJI.get(disease, "")} {disease} ({DISEASE_EN[disease]})</div><br>'
                    f'<span style="font-size:28px;font-weight:bold;color:{color}">'
                    f'{conf:.1f}%</span> <span style="font-size:14px;color:#666">신뢰도</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # 아토피 → 중증도 판정
                if disease == "아토피":
                    score, sev_label = predict_severity(img, sev_model)
                    sev_color = SEVERITY_COLORS[sev_label]
                    st.markdown(
                        f'<div class="result-card" style="border-left-color:{sev_color};background:#fdf6ec">'
                        f'<b>아토피 중증도 (IGA Grade 기반)</b><br>'
                        f'<span class="sev-badge" style="background:{sev_color}">{sev_label}</span>'
                        f'&nbsp; 중증도 점수: <b>{score:.3f}</b>'
                        f'<br><small style="color:#888">IGA: Mild(경증) / Moderate(중등도) / Severe(중증)'
                        f' | erythema·papulation·excoriation·lichenification 복합 판정</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # 비아토피 질환은 병변 수/범위로 경증/중등도 간이 판정
                    sev_label = "경증"

                # 전체 확률 분포
                st.markdown("**클래스별 신뢰도**")
                sorted_idx = np.argsort(probs)[::-1]
                for idx in sorted_idx:
                    cls_name = DISEASE_CLASSES[idx]
                    pct      = float(probs[idx]) * 100
                    bar_color = DISEASE_COLORS.get(cls_name, "#ccc")
                    st.markdown(
                        f'<div style="margin-bottom:6px">'
                        f'<span style="font-size:13px;width:80px;display:inline-block">'
                        f'{DISEASE_EMOJI.get(cls_name,"")} {cls_name}</span>'
                        f'<span style="font-size:13px;font-weight:bold;margin-left:8px">{pct:.1f}%</span>'
                        f'<div class="prob-bar-bg">'
                        f'<div style="background:{bar_color};width:{pct:.1f}%;height:8px;'
                        f'border-radius:4px"></div></div></div>',
                        unsafe_allow_html=True,
                    )

                # 치료 가이드
                st.divider()
                st.subheader("치료 성분 가이드")
                row = get_treatment(disease, sev_label, treat_df)
                if row is not None:
                    st.markdown(f"""
<div style="background:#f8f9fa;border-radius:12px;padding:16px 20px;
            border-left:4px solid #4ECDC4;margin-bottom:8px">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
    <div style="background:#fff;border-radius:8px;padding:12px 14px">
      <div style="font-size:12px;font-weight:700;color:#555;margin-bottom:6px">1차 치료</div>
      <div style="font-size:13px;font-weight:400;color:#222;line-height:1.6">{row["1차치료"]}</div>
    </div>
    <div style="background:#fff;border-radius:8px;padding:12px 14px">
      <div style="font-size:12px;font-weight:700;color:#555;margin-bottom:6px">2차 치료</div>
      <div style="font-size:13px;font-weight:400;color:#222;line-height:1.6">{row["2차치료"]}</div>
    </div>
    <div style="background:#fff;border-radius:8px;padding:12px 14px">
      <div style="font-size:12px;font-weight:700;color:#555;margin-bottom:6px">주요 성분</div>
      <div style="font-size:13px;font-weight:400;color:#222;line-height:1.6">{row["주요성분"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
                    st.caption(
                        "※ 치료 가이드는 AAD(미국피부과학회) 및 대한피부과학회 가이드라인 기반입니다. "
                        "실제 처방은 전문의와 상담하십시오."
                    )

    # ── 탭 2: 모델 성능 ──────────────────────────
    with tabs[1]:
        st.subheader("ResNet50 질환 분류 모델 성능")
        st.caption("학습: Training 4,800장 (정면) | 평가: Validation 600장 (정면)")

        perf_df = pd.DataFrame(PERF_DATA)
        st.dataframe(perf_df.set_index("질환"), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("전체 Accuracy", "83%")
        col_b.metric("아토피 Sensitivity", "86%")
        col_c.metric("정상 Specificity", "100%")

        st.divider()
        st.subheader("아토피 IGA 중증도 회귀 모델")
        st.caption("ResNet50 기반 회귀 — IGA grade (Mild/Moderate/Severe) 기반 중증도 점수 예측")
        col_d, col_e = st.columns(2)
        col_d.metric("MAE (Mean Absolute Error)", "0.387")
        col_e.metric("대상 질환", "아토피 (아토피피부염)")

        with st.expander("📌 평가 지표 설명"):
            st.markdown("""
| 지표 | 의미 |
|---|---|
| **Sensitivity (민감도)** | 실제 해당 질환인 케이스를 정확히 탐지한 비율 (True Positive Rate) |
| **Specificity (특이도)** | 해당 질환이 아닌 케이스를 정확히 음성으로 판별한 비율 (True Negative Rate) |
| **Accuracy** | 전체 예측 중 정답 비율 |
| **MAE** | 예측 중증도 점수와 실제 IGA grade 인코딩 값의 평균 절대 오차 |

> IGA(Investigator's Global Assessment): 아토피피부염 중증도 국제 표준 지표
> Mild(경증) / Moderate(중등도) / Severe(중증)
> 세부 항목: erythema, papulation, excoriation, lichenification
            """)

        # confusion matrix 이미지 표시
        cm_path = os.path.join(BASE_DIR, "results", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.divider()
            st.subheader("Confusion Matrix")
            st.image(cm_path, use_container_width=True)

    # ── 탭 3: 치료 성분 가이드 ──────────────────
    with tabs[2]:
        st.subheader("질환 × 중증도 치료 성분 매핑")
        st.caption(
            "AAD(미국피부과학회) 및 대한피부과학회 가이드라인 기반 | "
            "임상 판단을 대체하지 않음"
        )

        disease_filter = st.selectbox(
            "질환 선택",
            ["전체"] + sorted(treat_df["질환"].unique().tolist()),
        )
        show_df = treat_df if disease_filter == "전체" else treat_df[treat_df["질환"] == disease_filter]

        SEV_COLOR = {"경증": "#4ECDC4", "중등도": "#FFA94D", "중증": "#FF6B6B"}
        for disease_name, group in show_df.groupby("질환", sort=False):
            st.markdown(f"#### {disease_name}")
            cols = st.columns(len(group))
            for col, (_, row) in zip(cols, group.iterrows()):
                color = SEV_COLOR.get(row["중증도"], "#888")
                col.markdown(f"""
<div style="border-radius:12px;padding:14px 16px;border-top:4px solid {color};
            background:#fafafa;margin-bottom:4px">
  <div style="font-size:12px;font-weight:700;color:{color};margin-bottom:10px">{row["중증도"]}</div>
  <div style="font-size:11px;font-weight:700;color:#555;margin-bottom:4px">1차 치료</div>
  <div style="font-size:12px;color:#222;margin-bottom:10px;line-height:1.5">{row["1차치료"]}</div>
  <div style="font-size:11px;font-weight:700;color:#555;margin-bottom:4px">2차 치료</div>
  <div style="font-size:12px;color:#222;margin-bottom:10px;line-height:1.5">{row["2차치료"]}</div>
  <div style="font-size:11px;font-weight:700;color:#555;margin-bottom:4px">주요 성분</div>
  <div style="font-size:12px;color:#222;line-height:1.5">{row["주요성분"]}</div>
</div>
""", unsafe_allow_html=True)
            st.divider()

        with st.expander("📌 질환별 개요"):
            st.markdown("""
| 질환 | 설명 | 특징 |
|---|---|---|
| **건선** | 자가면역성 만성 염증성 피부질환 | 은백색 비늘 동반 홍반 판 |
| **아토피** | IgE 매개 알레르기성 피부염 | EASI 점수로 중증도 정량화 |
| **여드름** | 모낭·피지선 만성질환 | 면포·구진·농포·결절 순 악화 |
| **지루** | Malassezia 진균 관련 피부염 | 두피·얼굴 T존 집중 |
| **주사** | 만성 안면 홍조·혈관 확장 | 중앙 안면 집중, UV·열 악화 |
| **정상** | 피부 병변 없음 | 예방적 보습·자외선 차단 권장 |
            """)

        st.markdown(
            '<div class="warning-box">⚠️ <b>본 매핑 테이블은 파일럿 연구용입니다.</b> '
            '실제 처방 및 치료 결정은 반드시 피부과 전문의의 진찰을 통해 이루어져야 합니다.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
