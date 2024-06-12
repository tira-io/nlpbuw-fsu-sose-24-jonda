from pathlib import Path
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def textrank_sentences(sentences, top_n=3):
    if not sentences:
        return "No content available."

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    sim_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [sentence for score, sentence in ranked_sentences[:top_n]]
    summary = ' '.join(top_sentences)
    return summary

def summarize_story(story, top_n=3):
    sentences = preprocess_text(story)
    return textrank_sentences(sentences, top_n)

if __name__ == "__main__":
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    df["summary"] = df["story"].apply(lambda x: summarize_story(x, top_n=3))
    df = df.drop(columns=["story"]).reset_index()

    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
    print("Predictions saved to", Path(output_directory) / "predictions.jsonl")
