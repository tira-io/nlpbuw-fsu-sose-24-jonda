from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

model = BertForSequenceClassification.from_pretrained("/code/model")
tokenizer = BertTokenizer.from_pretrained("/code/model")

tira = Client()
df = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")

df = df.sample(frac=0.1, random_state=42)

def encode_sentences(sentence1, sentence2):
    return tokenizer(sentence1, sentence2, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

encoded_data = df.apply(lambda row: encode_sentences(row['sentence1'], row['sentence2']), axis=1)

input_ids = torch.cat([item['input_ids'] for item in encoded_data.values])
attention_masks = torch.cat([item['attention_mask'] for item in encoded_data.values])

model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)
    predictions = torch.argmax(outputs.logits, dim=1)

df["label"] = predictions.numpy()
df = df.drop(columns=["sentence1", "sentence2"]).reset_index()

output_directory = get_output_directory(str(Path(__file__).parent))
df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)

truth = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")

results = df.set_index("id").join(truth)

accuracy = accuracy_score(results["label"], results["truth"])
print(f"Accuracy: {accuracy:.4f}")