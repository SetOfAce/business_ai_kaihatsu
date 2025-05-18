from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

# モデル名
model_name = "skylord/swin-finetuned-food101"

# モデルと特徴量抽出器を読み込み
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# 分類対象の画像を読み込む
image = Image.open("チャーハン.jpg").convert("RGB")  # 画像ファイルのパスを指定

# 前処理
inputs = feature_extractor(images=image, return_tensors="pt")

# 推論
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_id]

print(f"予測された料理: {predicted_label}")





import sqlite3

def get_kcal_by_name(menu_name):
    # データベースに接続（ファイル名は適宜変更）
    conn = sqlite3.connect("カロリー表.db")
    cursor = conn.cursor()

    # nameが一致する行のkcalを取得
    cursor.execute("SELECT kcal FROM カロリー表 WHERE name = ?", (menu_name,))
    result = cursor.fetchone()

    # 接続を閉じる
    conn.close()

    # 結果の処理
    if result:
        return result[0]  # kcal値
    else:
        return None       # 一致しない場合

# 使用例
menu = predicted_label
kcal = get_kcal_by_name(menu)
if kcal is not None:
    print(f"{menu} のカロリーは {kcal} kcal です。")
else:
    print(f"{menu} はデータベースに存在しません。")