from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
import re

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    person_names = {"Alexander", "John", "Mary", "Lukashenko"}
    location_names = {"Belarus", "U.S.", "Germany"}

    location_suffixes = ["stan", "land", "ia", "is", "as"]
    organization_suffixes = ["Inc", "Ltd", "Corp", "Group", "Co"]


    def heuristic_ner_predictor(sentence):
        tokens = sentence.split()
        tags = ['O'] * len(tokens)

        for i, token in enumerate(tokens):
            if token in person_names:
                tags[i] = 'B-per'
            elif token in location_names:
                tags[i] = 'B-geo'
            elif any(token.endswith(suffix) for suffix in organization_suffixes):
                tags[i] = 'B-org'
            elif any(token.endswith(suffix) for suffix in location_suffixes):
                tags[i] = 'B-geo'
            elif token[0].isupper() and re.match(r'^[A-Z][a-z]+$', token):
                tags[i] = 'B-geo'
        return tags

    # Labeling the data
    predictions = text_validation.copy()
    predictions['tags'] = predictions['sentence'].apply(heuristic_ner_predictor)
    predictions = predictions[['id', 'tags']]

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
