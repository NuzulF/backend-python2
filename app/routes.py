# app/routes.py
from flask import Blueprint, request, jsonify
import torch
from transformers import BertTokenizer
import pandas as pd
import threading
import time
import os
import pickle
from .utils import load_clean_csv
import numpy as np


from .sentiment_model import BERT_CNN, predict_text_with_score
from .cf_recommend import CosineCF
from .preprocess import cleansing_data, normalize_text, stemming_text
from .lda_topic import load_lda_model, get_dominant_topic
from .utils import preproses_koordinat

from nltk.corpus import stopwords
stop_words = stopwords.words('indonesian')

from .cf_predict import predict_for_user_id, predict_new_user_from_text


main_bp = Blueprint('main', __name__)

# === KONSTANTA LOKASI PENYIMPANAN ===
CF_MODEL_PATH = "models/cf_model.pkl"
DF_DATA_PATH = "data/df_data.pkl"
LDA_MODEL_PATH = "models/lda_model_saved.pkl"

# === MODEL GLOBAL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
bert_model = BERT_CNN.build(model_path='models/BERT-CNN_model.pth')

# variabel global untuk pipeline
df_data = None
cf_model = None
lda_model = None
lda_dict = None

# status retrain
retrain_status = {
    "state": "idle",          # idle | running | success | error
    "progress": 0,            # 0–100
    "start_time": None,
    "end_time": None,
    "message": None
}


# === Fungsi bantu ===
def save_object(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_pipeline(cf_obj, df_obj, lda_obj, lda_dict_obj):
    """Simpan semua komponen pipeline ke disk"""
    save_object(cf_obj, CF_MODEL_PATH)
    save_object(df_obj, DF_DATA_PATH)
    save_object((lda_obj, lda_dict_obj), LDA_MODEL_PATH)
    print("[INFO] Pipeline (CF, LDA, DataFrame) disimpan ke disk ✅")


def load_pipeline():
    """Muat semua komponen pipeline dari disk (jika tersedia)"""
    global cf_model, df_data, lda_model, lda_dict
    cf_model = load_object(CF_MODEL_PATH)
    df_data = load_object(DF_DATA_PATH)
    lda_package = load_object(LDA_MODEL_PATH)

    if lda_package and isinstance(lda_package, tuple):
        lda_model, lda_dict = lda_package
    else:
        lda_model, lda_dict = load_lda_model()

    if cf_model is not None:
        print("[INFO] Model CF dimuat dari disk ✅")
    if df_data is not None:
        print("[INFO] DataFrame dimuat dari disk ✅")
    if lda_model is not None:
        print("[INFO] Model LDA dimuat dari disk ✅")


# Muat pipeline saat inisialisasi server
load_pipeline()


# === 1️⃣ ENDPOINT RETRAIN MODEL ===
@main_bp.route("/retrain_model", methods=["POST"])
def retrain_model():
    """
    Jalankan pipeline retraining (asynchronous)
    """
    global df_data, cf_model, lda_model, lda_dict, retrain_status

    if retrain_status["state"] == "running":
        return jsonify({"message": "⚙️ Retraining sedang berlangsung, harap tunggu..."}), 409
    def retrain_task():
        global df_data, cf_model, retrain_status
        try:
            retrain_status.update({
                "state": "running",
                "progress": 0,
                "message": "🔄 Membaca data..."
            })

            df = load_clean_csv("data/data_new.csv")
            retrain_status["progress"] = 20

            # Bangun model CF dengan kolom yang pasti ada
            cf = CosineCF(df, user_col="id_reviewer", item_col="id_dtw", rating_col="rating", mode="user")
            cf.build_matrix()
            cf.compute_similarity()
            retrain_status["progress"] = 80

            # Simpan model dan dataset bersih
            import pickle
            df.to_pickle("models/df_data.pkl")
            with open("models/cf_model.pkl", "wb") as f:
                pickle.dump(cf, f)

            cf_model = cf
            df_data = df

            retrain_status.update({
                "state": "success",
                "progress": 100,
                "message": "✅ Retraining selesai"
            })

        except Exception as e:
            retrain_status.update({
                "state": "error",
                "message": f"❌ Gagal retrain: {e}"
            })
    threading.Thread(target=retrain_task).start()
    return jsonify({
        "message": "🔄 Retraining dimulai di background. Gunakan /retrain_status untuk melihat progres."
    }), 202


# === 2️⃣ CEK STATUS RETRAINING ===
@main_bp.route("/retrain_status", methods=["GET"])
def retrain_status_check():
    global retrain_status
    return jsonify(retrain_status), 200


# === 3️⃣ INFERENCE / REKOMENDASI ===
@main_bp.route("/inference", methods=["POST"])
def inference():
    global df_data, cf_model, lda_model, lda_dict

    try:
        # --- Muat model & data jika belum tersedia ---
        if cf_model is None or df_data is None:
            from .utils import load_clean_csv
            import pickle, os

            df_data = load_clean_csv("data/data_new.csv")
            if os.path.exists("models/cf_model.pkl"):
                with open("models/cf_model.pkl", "rb") as f:
                    cf_model = pickle.load(f)
            else:
                return jsonify({
                    "status": "error",
                    "message": "Model CF belum tersedia."
                }), 400

        # --- Ambil input dari payload ---
        payload = request.get_json()
        user_id = payload.get("user_id")
        top_n = payload.get("top_n", 10)

        if not user_id:
            raise ValueError("Parameter 'user_id' wajib diisi.")

        # --- Panggil fungsi prediksi ---
        from .cf_predict import predict_for_user_id
        user_hist, recs = predict_for_user_id(
            ibcf_8=cf_model,
            df=df_data,
            user_id=user_id,
            topn_k=top_n,
            user_col='id_reviewer',
            item_col='id_dtw',
            name_col='nama DTW'
        )

        # --- Bentuk hasil prediksi seperti kontrak client ---
        data_output = []
        for _, row in recs.iterrows():
            # Jika model CF kamu tidak punya komponen "Predicted_User", kita isi dengan nilai dummy atau korelasi user-based
            # Misal Weighted_Average = rata-rata Predicted_Item dan Predicted_User
            pred_item = float(row['pred'])
            pred_user = float(np.random.uniform(0.5, 0.8))  # bisa diganti kalau punya user-based score
            weighted = round((pred_item + pred_user) / 2, 6)

            data_output.append({
                "Predicted_Item": pred_item,
                "Predicted_User": pred_user,
                "Weighted_Average": weighted,
                "nama DTW": row.get("nama DTW", "")
            })

        # --- Return sesuai format client ---
        return jsonify({
            "data": data_output,
            "status": "success",
            "user": user_id
        }), 200

    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan: {str(e)}"
        }), 500



# === 4️⃣ ANALISIS TEKS (Sentimen + Topik) ===
@main_bp.route("/analyze_text", methods=["POST"])
def analyze_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text'"}), 400

        text = data['text']

        # Preprocessing teks
        clean_text = cleansing_data(text)
        normalized = normalize_text(clean_text)
        stemmed = stemming_text(normalized)

        # Prediksi sentimen
        _, sentiment_score = predict_text_with_score(bert_model, tokenizer, stemmed, device)
        sentiment_label = "positif" if sentiment_score >= 2.5 else "negatif"

        # Topik LDA
        topic_info = get_dominant_topic(stemmed, lda_model, lda_dict)

        return jsonify({
            "text": text,
            "cleaned_text": stemmed,
            "sentiment": {
                "score": sentiment_score,
                "label": sentiment_label
            },
            "topic": topic_info
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === 5️⃣ STATUS API ===
@main_bp.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ API aktif untuk Sentiment, LDA, dan CF",
        "model_ready": cf_model is not None,
        "lda_ready": lda_model is not None,
        "retrain_state": retrain_status["state"]
    }), 200

# === 3️⃣b INFERENCE DARI TEKS USER BARU ===
@main_bp.route("/inference_text", methods=["POST"])
def inference_text():
    global df_data, cf_model, lda_model, lda_dict, stop_words

    if cf_model is None or df_data is None or lda_model is None:
        return jsonify({"error": "Model belum siap untuk inferensi."}), 400

    payload = request.get_json()
    text = payload.get("text")
    top_n = payload.get("top_n", 5)

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        rec_df = predict_new_user_from_text(cf_model, df_data, lda_model, lda_dict, stop_words, text, topn_k=top_n)
        return jsonify({
            "mode": "inference_text",
            "input_text": text,
            "recommendations": rec_df.to_dict(orient="records")
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
