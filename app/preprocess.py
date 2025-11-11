# app/preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

kamus = pd.read_csv('data/kata_tidak_baku.csv')
mapping = dict(zip(kamus['kata_tidak_baku'], kamus['kata_baku']))

def case_folding(text):
    return text.lower() if isinstance(text, str) else text

def remove_URL(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text) if isinstance(text, str) else text

def remove_html(text):
    return re.sub(r'<.*?>', '', text) if isinstance(text, str) else text

def remove_emoji(tweet):
    if tweet is not None and isinstance(tweet, str):
        import re
        emoji_pattern = re.compile("[" 
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F700-\U0001F77F"
            u"\U0001F780-\U0001F7FF"
            u"\U0001F800-\U0001F8FF"
            u"\U0001F900-\U0001F9FF"
            u"\U0001FA00-\U0001FA6F"
            u"\U0001FA70-\U0001FAFF"
            u"\U0001F004-\U0001F0CF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', tweet)
    return tweet

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text) if isinstance(text, str) else text

def remove_numbers(text):
    return re.sub(r'\d+', '', text) if isinstance(text, str) else text

def remove_username(text):
    return re.sub(r'@[^\s]+', '', text) if isinstance(text, str) else text

def remove_apostrof(text):
    return re.sub(r"[’']", '', text) if isinstance(text, str) else text

def cleansing_data(text):
    text = case_folding(text)
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_symbols(text)
    text = remove_numbers(text)
    text = remove_username(text)
    return text

def normalize_text(text):
    if isinstance(text, str):
        words = text.split()
        return ' '.join([mapping.get(word, word) for word in words])
    return text

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming_text(text):
    return stemmer.stem(text) if isinstance(text, str) else text

def preprocess_text(text, stop_words):
    tambahan_stopword = [
        'menjadi','sangat','banget','mau','jadi','buat','yg','jd','bs','dg','dgn','tp','th','km','k',
        'sgt','nya','utk','cukup','masuk','kalo','kalau','banyak','tempat','and','merupakan','the',
        'to','was','in','we','ga','gak','disana','what','ya','pas','penuh','dll','lebih','kurang',
        'our','benar','atas','menikmati','kesini'
    ]
    stop_words = stop_words + tambahan_stopword
    return [word for word in simple_preprocess(text) if word not in stop_words]

def kosong_to_null(text):
    if isinstance(text, str) and text.strip().lower() == 'kosong':
        return ''
    return text

def pra_proses_dan_stemming(df):
    df_clean = df.copy()
    df_clean['review'] = df_clean['review'].apply(kosong_to_null)
    df_clean['deskripsi'] = df_clean['deskripsi'].apply(kosong_to_null)

    df_clean['review_clean'] = df_clean['review'].apply(cleansing_data)
    df_clean['deskripsi_clean'] = df_clean['deskripsi'].apply(cleansing_data)

    df_clean['review_normalized'] = df_clean['review_clean'].apply(normalize_text).apply(remove_apostrof)
    df_clean['deskripsi_normalized'] = df_clean['deskripsi_clean'].apply(normalize_text).apply(remove_apostrof)

    df_clean['review_stemmed'] = df_clean['review_normalized'].apply(stemming_text)
    df_clean['deskripsi_stemmed'] = df_clean['deskripsi_normalized'].apply(stemming_text)

    df_clean['review_tokens'] = df_clean['review_stemmed'].apply(lambda x: preprocess_text(x, stop_words))
    df_clean['deskripsi_tokens'] = df_clean['deskripsi_stemmed'].apply(lambda x: preprocess_text(x, stop_words))
    return df_clean
