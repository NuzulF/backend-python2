# app/cf_recommend.py
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class CosineCF:
    def __init__(self, df, user_col, item_col, rating_col, mode="user"):
        self.df = df.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.mode = mode
        self.similarity_matrix = None
        self.user_item_matrix = None

    def build_matrix(self):
        self.user_item_matrix = self.df.pivot_table(
            index=self.user_col if self.mode == "user" else self.item_col,
            columns=self.item_col if self.mode == "user" else self.user_col,
            values=self.rating_col
        ).fillna(0)
        return self.user_item_matrix

    def compute_similarity(self):
        if self.user_item_matrix is None:
            self.build_matrix()
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        return pd.DataFrame(
            self.similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )



    def recommend(self, target_id, top_n=5):
        """
        Recommend top_n items for target_id (user or item depending on self.mode).
        Returns list of dicts: [{item_col: ..., "score_norm": ...}, ...]
        """
        # pastikan similarity matrix ada
        if self.similarity_matrix is None:
            self.compute_similarity()

        # pastikan target ada di index
        if target_id not in self.user_item_matrix.index:
            return []

        # ambil index (posisi) target
        target_pos = self.user_item_matrix.index.get_loc(target_id)

        # ambil vector similarity (pastikan numpy 1D)
        sim_scores = np.asarray(self.similarity_matrix[target_pos]).ravel()

        # buat dataframe pairing id <-> similarity
        sim_df = pd.DataFrame({
            "id": list(self.user_item_matrix.index),
            "similarity": sim_scores
        })

        # urutkan berdasarkan similarity descending
        sim_df = sim_df.sort_values(by="similarity", ascending=False).reset_index(drop=True)

        # ambil top N + skip self (pos 0 biasanya adalah self)
        # gunakan top_n parameter (ambil top_n paling dekat selain self)
        # jika jumlah baris kecil, sesuaikan batasnya
        # cari baris yang bukan target_id kemudian ambil top_n
        sim_df = sim_df[sim_df["id"] != target_id]
        top_similar_ids = sim_df.head(top_n)["id"].tolist()

        # jika mode user: temukan item yang dikonsumsi oleh top_similar users tetapi belum dikonsumsi target user
        if self.mode == "user":
            # data pengguna yang mirip
            other_users = self.df[self.df[self.user_col].isin(top_similar_ids)]

            # items yang sudah dilihat target
            target_items = set(self.df[self.df[self.user_col] == target_id][self.item_col].unique())

            # filter item yang belum dibaca target
            unseen_items = other_users[~other_users[self.item_col].isin(target_items)]

            if unseen_items.empty:
                return []

            rec_df = unseen_items.groupby(self.item_col)[self.rating_col].mean().reset_index()
        else:
            # mode item: treat top_similar_ids as similar items
            similar_items = top_similar_ids
            rec_df = self.df[self.df[self.item_col].isin(similar_items)].groupby(self.item_col)[self.rating_col].mean().reset_index()
            if rec_df.empty:
                return []

        # normalisasi score (cek ukuran sebelum fit)
        try:
            scaler = MinMaxScaler()
            # jika hanya satu row, fit_transform tetap menghasilkan array 1x1
            rec_df["score_norm"] = scaler.fit_transform(rec_df[[self.rating_col]])
        except Exception:
            # fallback: gunakan nilai rating kolom langsung jika scaling gagal
            rec_df["score_norm"] = rec_df[self.rating_col]

        # urutkan dan ambil top_n
        rec_df = rec_df.sort_values(by="score_norm", ascending=False).head(top_n)

        # kembalikan list dict
        return rec_df[[self.item_col, "score_norm"]].rename(columns={self.item_col: self.item_col}).to_dict(orient="records")
        
