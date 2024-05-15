from pathlib import Path
from tqdm import tqdm
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from collections import Counter

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

    def generate_ngrams(text, n):
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i + n])
        return ngrams


    def process_stopwords(folder_path, languages, max_n=3):
        ngram_counts_per_language = {}
        print("Processing stopwords ...")
        for lang in tqdm(languages):
            stop_words_path = folder_path / f"stopwords-{lang}.txt"
            if not stop_words_path.exists():
                print(f"Stopwords file not found for language: {lang}")
                continue

            with open(stop_words_path, "r", encoding="utf-8") as file:
                stopwords = file.read().splitlines()

            ngram_counts = Counter()
            for n in range(1, max_n + 1):
                for stop_word in stopwords:
                    ngram_counts.update(generate_ngrams(stop_word, n))

            ngram_counts_per_language[lang] = ngram_counts

        return ngram_counts_per_language
    
    def predict_languages(text_data, stopwords_folder, output_file, ngram_counts_per_language):
        predictions = []

        text_data_size = len(text_data)
        print("Predicting languages ...")
        for _, example in tqdm(text_data.iterrows(), total=text_data_size):
            text = example["text"]
            text_ngrams = Counter()
            for lang, ngram_counts in ngram_counts_per_language.items():
                for ngram, count in ngram_counts.items():
                    text_ngrams[ngram] += text.count(ngram)

            predicted_lang = max(ngram_counts_per_language.keys(), key=lambda lang: sum(text_ngrams[ngram] for ngram in ngram_counts_per_language[lang]))
            predictions.append({"id": example["id"], "lang": predicted_lang})

        output_file = get_output_directory(str(Path(__file__).parent))
        predictions = pd.DataFrame(predictions)
        predictions.to_json(
            Path(output_file) / "predictions.jsonl", orient="records", lines=True
        )

    def calculate_accuracy(predictions_file, targets_validation):
        predictions = pd.read_json(predictions_file, lines=True)
        predictions = predictions.set_index("id")
        targets_validation = targets_validation.set_index("id")
        merged = predictions.join(targets_validation, how="inner")
        correct = (merged["lang"] == merged["lang_id"]).sum()
        total = len(merged)
        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    #folder_path = Path("./language-identification-stopwords/stopwords/")
    folder_path = Path(__file__).parent / "stopwords"

    ngram_counts_per_language = process_stopwords(folder_path, lang_ids)
    output_file = get_output_directory(str(Path(__file__).parent)) + "/predictions.jsonl"
    predict_languages(text_validation, folder_path, output_file, ngram_counts_per_language)