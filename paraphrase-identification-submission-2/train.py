from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import pandas as pd
from tira.rest_api_client import Client
from datasets import Dataset, DatasetDict

# taken from https://huggingface.co/transformers/v3.0.2/model_doc/bert.html

tira = Client()
text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
data = text.join(labels).reset_index()

data = data.sample(frac=0.1, random_state=42)  # Use 10% of the data for quick testing

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_sentences(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)

encoded_data = Dataset.from_pandas(data)
encoded_data = encoded_data.map(encode_sentences, batched=True)

encoded_data = encoded_data.rename_column("label", "labels")
encoded_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_val_split = encoded_data.train_test_split(test_size=0.1)
train_data = train_val_split['train']
val_data = train_val_split['test']

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=1,
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    warmup_steps=100,            
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

model.save_pretrained("/code/model")
tokenizer.save_pretrained("/code/model")
