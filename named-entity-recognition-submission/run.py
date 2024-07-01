from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    def simple_ner_predictor(sentence):
        tokens = sentence.split()
        tags = ['O'] * len(tokens)
        for i, token in enumerate(tokens):
            if token.istitle():
                tags[i] = 'B-per'  # Naive approach: label all title-case words as persons
        return tags

    # labeling the data
    predictions = text_validation.copy()
    predictions['tags'] = predictions['sentence'].apply(simple_ner_predictor)
    predictions = predictions[['id', 'tags']]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
