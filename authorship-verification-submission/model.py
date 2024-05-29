from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

if __name__ == "__main__":

    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    df = text.join(labels.set_index("id"))

    model = Pipeline(
        [("vectorizer", CountVectorizer()), ("classifier", SVC(kernel='linear'))]
    )
    model.fit(df["text"], df["generated"])

    dump(model, Path(__file__).parent / "model.joblib")