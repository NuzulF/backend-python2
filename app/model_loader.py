import joblib

def load_cf_model(path="models/cf_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] CF model not found: {e}")
        return None

def load_lda_model(path="models/lda_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] LDA model not found: {e}")
        return None
