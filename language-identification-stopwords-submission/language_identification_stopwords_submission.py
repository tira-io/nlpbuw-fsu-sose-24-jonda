from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":    

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    
    lang_ids = [
        "af",
        "az",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "hr",
        "it",
        "ko",
        "nl",
        "no",
        "pl",
        "ru",
        "ur",
        "zh",
    ]

    print("preprocessing text data")
    text_data = text_validation.iloc[:, 1].tolist()
    target_labels = targets_validation.iloc[:, 1].tolist()


    print("Extract features using TF-IDF")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(text_data)

    print("Training the classifier")
    svc_classifier = SVC(verbose=True)
    svc_classifier.fit(tfidf_vectors, target_labels)

    print("Predicting the validation data")
    predictions = svc_classifier.predict(tfidf_vectors)

    print("Evaluating the classifier")
    accuracy = (predictions == target_labels).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("Saving the predictions")
    output_file = get_output_directory(str(Path(__file__).parent)) + "/predictions.jsonl"
    predictions_df = pd.DataFrame({"id": text_validation["id"], "lang": predictions})
    predictions_df.to_json(output_file, orient="records", lines=True)
