from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import pandas as pd
from tira.rest_api_client import Client
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# Load data
tira = Client()
text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
data = text.join(labels).reset_index()

# Use a subset of the data
data = data.sample(frac=0.1, random_state=42)  # Use 10% of the data for quick testing

# Preprocess data
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def encode_sentences(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)

encoded_data = Dataset.from_pandas(data)
encoded_data = encoded_data.map(encode_sentences, batched=True)

# Rename 'label' column to match expected 'labels' column name
encoded_data = encoded_data.rename_column("label", "labels")
encoded_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split data into train and validation sets
train_val_split = encoded_data.train_test_split(test_size=0.1)
train_data = train_val_split['train']
val_data = train_val_split['test']

# Define Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Fine-tune ALBERT model
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=1,  # Reduce number of epochs for quick testing
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    warmup_steps=100,  # Reduce warmup steps for quick testing              
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_data,         
    eval_dataset=val_data,
    data_collator=data_collator,
    tokenizer=tokenizer                  
)

trainer.train()

# Save the model
model.save_pretrained("/code/model")
tokenizer.save_pretrained("/code/model")
