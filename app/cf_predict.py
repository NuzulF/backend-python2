import pandas as pd
import numpy as np
from .lda_topic import get_dominant_topic

# Untuk normalisasi skor
def _minmax01(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ===========================================================
# 1️⃣ PREDIKSI UNTUK USER YANG SUDAH ADA (BERDASARKAN USER_ID)
# ===========================================================
def predict_for_user_id(ibcf_8, df, user_id,
                        topn_k=5,
                        user_col='id_reviewer',
                        item_col='id_dtw',
                        name_col='nama DTW'):
    id2name = {}
    if name_col in df.columns:
        id2name = (df[[item_col, name_col]]
                     .dropna()
                     .drop_duplicates(subset=[item_col])
                     .set_index(item_col)[name_col]
                     .to_dict())

    hist = df[df[user_col] == user_id].copy()
    rated_items = set(hist[item_col].unique())
    show_cols = [item_col, ibcf_8.rating_col]
    if name_col in df.columns:
        show_cols.insert(1, name_col)
    hist = hist[show_cols].dropna().drop_duplicates(subset=[item_col])
    hist = hist.rename(columns={ibcf_8.rating_col: 'user_score'})
    if 'user_score' in hist.columns:
        hist['user_score'] = hist['user_score'].round(3)

    # jika user sudah ada dalam model (punya mapping u2i)
    if hasattr(ibcf_8, "u2i") and user_id in ibcf_8.u2i:
        rec = ibcf_8.recommend_for_user(user_id, n=topn_k, candidates='unseen')
    else:
        # cold start → populer by model target
        pop = (df.groupby(item_col)[ibcf_8.rating_col].mean()
                 .sort_values(ascending=False).reset_index()
                 .rename(columns={ibcf_8.rating_col: 'pred'}))
        if rated_items:
            pop = pop[~pop[item_col].isin(rated_items)]
        rec = pop.head(topn_k)

    if name_col in df.columns and len(rec):
        rec[name_col] = rec[item_col].map(id2name)
    rec = rec.copy()
    rec['pred'] = rec['pred'].round(3)

    rec_cols = [item_col, 'pred']
    if name_col in df.columns:
        rec_cols.insert(1, name_col)
    rec = rec[rec_cols]

    return hist.reset_index(drop=True), rec.reset_index(drop=True)


# ===========================================================
# 2️⃣ PREDIKSI UNTUK USER BARU (DARI TEKS)
# ===========================================================
def predict_new_user_from_text(ibcf_8, df, lda_model, dictionary, stop_words, text,
                               item_col='id_dtw', name_col='nama DTW', desc_col='deskripsi',
                               topn_k=10, beta=0.8):
    """
    New user types a description -> find dominant topic -> recommend items
    by blending topic match and item popularity from ibcf_8.
    """

    # 1️⃣ dominant topic dari teks user
    topic_info = get_dominant_topic(text, lda_model, dictionary)
    u_topic = topic_info["topic_id"]
    u_score = topic_info["score"]

    # 2️⃣ deskripsi tiap item
    items = (df[[item_col, name_col, desc_col]]
             .drop_duplicates(subset=[item_col])
             .dropna(subset=[desc_col])
             .reset_index(drop=True))
    if items.empty:
        pop = (df.groupby(item_col)[ibcf_8.rating_col].mean()
                 .sort_values(ascending=False)
                 .reset_index()
                 .rename(columns={ibcf_8.rating_col: 'score'}))
        if name_col in df.columns:
            id2name = df.drop_duplicates(item_col).set_index(item_col)[name_col]
            pop[name_col] = pop[item_col].map(id2name)
        return pop.head(topn_k)

    # 3️⃣ dominant topic untuk tiap deskripsi item
    topics_scores = items[desc_col].apply(lambda s: get_dominant_topic(s, lda_model, dictionary))
    items['item_topic'] = topics_scores.apply(lambda t: t["topic_id"])
    items['item_tscore'] = topics_scores.apply(lambda t: t["score"])
    items['topic_match'] = np.where(items['item_topic'] == u_topic, items['item_tscore'], 0.0)

    # 4️⃣ popularitas baseline
    pop_s = df.groupby(item_col)[ibcf_8.rating_col].mean()
    items['popularity'] = items[item_col].map(pop_s)

    # 5️⃣ blend skor (normalisasi 0-1)
    s1 = _minmax01(items['topic_match'].values)
    s2 = _minmax01(items['popularity'].values)
    items['score'] = beta * s1 + (1.0 - beta) * s2

    # 6️⃣ rank hasil
    cols = [item_col, name_col] if name_col in items.columns else [item_col]
    out = items[cols + ['topic_match', 'popularity', 'score']].sort_values('score', ascending=False)
    return out.head(topn_k).reset_index(drop=True)
