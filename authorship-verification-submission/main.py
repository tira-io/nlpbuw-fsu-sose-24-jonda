import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
from torchtext.data import get_tokenizer
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd

print("Hello")
tira = Client()
text_train = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
)
targets_train = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
)
# loading validation data (automatically replaced by test data when run on tira)
text_validation = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
)
targets_validation = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
)

tokenizer = get_tokenizer("basic_english")

def tokenize(text):
    return tokenizer(text)

TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)

train_examples = [
    (text_train.iloc[i].text, targets_train.iloc[i].id) for i in range(len(text_train))
]
train_df = pd.DataFrame(train_examples, columns=['text', 'label'])
train_df.to_csv('train_data.csv', index=False)

valid_examples = [
    (text_validation.iloc[i].text, targets_validation.iloc[i].id) for i in range(len(text_validation))
]
valid_df = pd.DataFrame(valid_examples, columns=['text', 'label'])
valid_df.to_csv('valid_data.csv', index=False)

fields = [('text', TEXT), ('label', LABEL)]
train_data, valid_data = TabularDataset.splits(
    path='', format='csv', train='train_data.csv', validation='valid_data.csv', fields=fields
)

TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        return self.fc(hidden[-1, :, :])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(TEXT.vocab)
embedding_dim = 300
hidden_dim = 128
output_dim = 2  # Two classes: 0 for human, 1 for AI
num_layers = 1

model = SimpleModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=num_layers)
model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=64,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device
)

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        text = batch.text
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


N_EPOCHS = 10
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'simple_model.pt')
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')


model.load_state_dict(torch.load('simple_model.pt'))
test_loss = evaluate(model, valid_iterator, criterion)
print(f'Test Loss: {test_loss:.3f}')
