import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch
import sqlite3

# モデル読み込み（キャッシュして毎回再読み込みしないようにする）
@st.cache_resource
def load_model():
    model_name = "skylord/swin-finetuned-food101"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return feature_extractor, model

# カロリーをDBから取得
def get_kcal_by_name(menu_name):
    conn = sqlite3.connect("カロリー表.db")
    cursor = conn.cursor()
    cursor.execute("SELECT kcal FROM カロリー表 WHERE name = ?", (menu_name,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Streamlit UI
st.title("🍽️ 食事画像から料理名とカロリーを判定するアプリ")

uploaded_file = st.file_uploader("食事の画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # モデル読み込み
    feature_extractor, model = load_model()

    # 前処理・推論
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_id]

    # 表示
    st.subheader("🍛 判定結果")
    st.write(f"**料理名：** {predicted_label}")

    kcal = get_kcal_by_name(predicted_label)
    if kcal is not None:
        st.write(f"**カロリー：** {kcal} kcal")
    else:
        st.write("⚠️ データベースにカロリー情報が見つかりませんでした。")