from pathlib import Path
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

nltk.download('punkt')

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def score_sentences(sentences):
    if not sentences:
        return np.array([])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).flatten()
    return sentence_scores

def summarize_story(story, top_n=2):
    sentences = preprocess_text(story)
    if not sentences:
        return "No content available."
    sentence_scores = score_sentences(sentences)
    if sentence_scores.size == 0:
        return "No significant content available."
    ranked_sentences = sorted(((score, sentence) for sentence, score in zip(sentences, sentence_scores)), reverse=True)
    top_sentences = [sentence for score, sentence in ranked_sentences[:top_n]]
    summary = ' '.join(top_sentences)
    return summary

if __name__ == "__main__":

    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    df["summary"] = df["story"].apply(lambda x: summarize_story(x, top_n=2))
    df = df.drop(columns=["story"]).reset_index()

    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
    print("Predictions saved to", Path(output_directory) / "predictions.jsonl")
