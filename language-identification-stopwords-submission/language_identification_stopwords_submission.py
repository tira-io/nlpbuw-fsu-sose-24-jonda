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
    print(targets_validation.head())
    print(targets_validation['lang'].value_counts())
    
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
    text_data = text_validation['text'].tolist()
    if 'lang' in targets_validation.columns:
        target_labels = targets_validation['lang'].tolist()
    else:
        target_label_column = targets_validation.columns.intersection(lang_ids).tolist()
        if not target_label_column:
            raise ValueError("No target label column found in the validation data.")
        target_labels = targets_validation[target_label_column[0]].tolist()

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
