from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    # Load pre-trained NER model and tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


    def transform_predictions(sentence):
        tokens = tokenizer.tokenize(sentence)
        ner_results = ner_pipeline(sentence)
        tags = ['O'] * len(tokens)

        for result in ner_results:
            word_tokens = tokenizer.tokenize(result['word'])
            for i, token in enumerate(word_tokens):
                if i == 0:
                    tags[result['index'] - 1] = f"B-{result['entity_group']}"
                else:
                    tags[result['index'] - 1] = f"I-{result['entity_group']}"

        return tags


    # labeling the data
    predictions = text_validation.copy()
    predictions['tags'] = predictions['sentence'].apply(transform_predictions)
    predictions = predictions[['id', 'tags']]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)