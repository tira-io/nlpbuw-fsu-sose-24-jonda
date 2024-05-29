import subprocess
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Step 1: Run the training script
subprocess.run(['python3', '/code/train.py'], check=True)

# Step 2: Load the trained model and tokenizer from the local directory
model = MobileBertForSequenceClassification.from_pretrained("/code/model", local_files_only=True)
tokenizer = MobileBertTokenizer.from_pretrained("/code/model", local_files_only=True)

# Step 3: Load the data
tira = Client()
df = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")

# Use a subset of the data for quick testing
df = df.sample(frac=0.1, random_state=42)

# Step 4: Preprocess the data
def encode_sentences(sentence1, sentence2):
    return tokenizer(sentence1, sentence2, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

encoded_data = df.apply(lambda row: encode_sentences(row['sentence1'], row['sentence2']), axis=1)

input_ids = torch.cat([item['input_ids'] for item in encoded_data.values])
attention_masks = torch.cat([item['attention_mask'] for item in encoded_data.values])

# Step 5: Make predictions
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)
    predictions = torch.argmax(outputs.logits, dim=1)

# Step 6: Prepare the results
df["predicted_label"] = predictions.numpy()  # Rename the prediction column to avoid conflict
df = df.drop(columns=["sentence1", "sentence2"]).reset_index()

# Save the predictions
output_directory = get_output_directory(str(Path(__file__).parent))
df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)

# Step 7: Load the ground truth labels
truth = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")

# Print column names to debug
print("Prediction DataFrame columns:", df.columns)
print("Truth DataFrame columns:", truth.columns)

# Merge predictions with ground truth
results = df.set_index("id").join(truth)

# Print results DataFrame columns to debug
print("Results DataFrame columns:", results.columns)

# Step 8: Calculate accuracy
accuracy = accuracy_score(results["label"], results["predicted_label"])  # Use the renamed column
print(f"Accuracy: {accuracy:.4f}")
