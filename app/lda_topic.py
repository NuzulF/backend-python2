# app/lda_topic.py
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

def load_lda_model():
    try:
        lda_model = LdaModel.load("models/lda_model_review.gensim")
        dictionary = corpora.Dictionary.load("models/dictionary.dict")
    except FileNotFoundError:
        print("⚠️ Warning: Model LDA tidak ditemukan, membuat dummy model.")
        dictionary = Dictionary(common_texts)
        corpus = [dictionary.doc2bow(text) for text in common_texts]
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
    return lda_model, dictionary


# def load_lda_model():
#     lda_model = LdaModel.load("models/lda_model_review.gensim")
#     dictionary = corpora.Dictionary.load("models/dictionary.dict")
#     return lda_model, dictionary

def get_dominant_topic(text, lda_model, dictionary):
    tokens = simple_preprocess(text)
    bow = dictionary.doc2bow(tokens)
    topic_distribution = lda_model.get_document_topics(bow)
    if not topic_distribution:
        return {"topic_id": None, "score": 0.0}
    topic_id, score = max(topic_distribution, key=lambda x: x[1])
    return {"topic_id": int(topic_id), "score": round(score, 4)}
