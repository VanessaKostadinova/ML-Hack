import csv
from random import Random
import gensim
import matplotlib.pyplot as plt
from gensim import downloader
import torch.optim
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import spacy

from encoder_fc_model import Encoder, Decoder, EncoderFC

nlp = spacy.load("en_core_web_sm")

positive_items = []
negative_items = []

print("Hello World")

with open("./dataset/2011Tornado_Summary.csv", "r", encoding="utf-8") as file:
    file.readline()
    for l in csv.reader(file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL):
        tokens = nlp(l[1])
        filtered_tokens = [token.lemma_ for token in tokens if not token.is_stop and not token.is_punct]
        item = [int(l[0]), filtered_tokens, l[2], int(l[3])]
        if item[3] == 1:
            positive_items.append(item)
        else:
            negative_items.append(item)

diff = abs(len(positive_items) - len(negative_items))

if len(positive_items) > len(negative_items):
    random = Random()
    for c in range(diff):
        del positive_items[(random.randint(0, len(positive_items) - 1))]
else:
    random = Random()
    for c in range(diff):
        del negative_items[(random.randint(0, len(negative_items) - 1))]

cutoff = int((len(negative_items) / 4) * 3)

training_set = negative_items[0:cutoff] + positive_items[0:cutoff]
testing_set = negative_items[cutoff + 1:] + positive_items[cutoff + 1:]


class BVHDataset(Dataset):
    def __init__(self):
        """self.items = []

        with open("./dataset/2011Tornado_Summary.csv", "r", encoding="utf-8") as file:
            file.readline()
            for l in csv.reader(file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL):
                tokens = nlp(l[1])
                filtered_tokens = [token.lemma_ for token in tokens if not token.is_stop and not token.is_punct]
                item = [int(l[0]), filtered_tokens, l[2], int(l[3])]
                print(item)
                self.items.append(item)"""

    def __len__(self):
        return len(training_set)

    def __getitem__(self, index):
        return [training_set[index][1], training_set[index][3]]


num_epochs = 19
batch_size = 1

load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = gensim.downloader.load('glove-wiki-gigaword-300')

num_joints = 22
hidden_size = 512
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

data = BVHDataset()

train_iterator = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, pin_memory=True)

encoder = Encoder(embedding_model, hidden_size, num_layers, encoder_dropout, device).to(device)
decoder = Decoder(hidden_size).to(device)

model = EncoderFC(encoder, decoder, hidden_size, device).to(device)
model_name = "./model_test.pt"

loss_fn = nn.CrossEntropyLoss().to(device)

optimiser = optim.Adam(model.parameters(), lr=0.001)
'''
# eval
model.load_state_dict(torch.load(model_name))
model.eval()

fp = 0
fn = 0

for item in testing_set:
    y_pred = model([[i] for i in item[1]])
    print(y_pred)
    print(item[3])
    if int(y_pred) != item[3]:
        if item[3] == 1:
            fp += 1
        else:
            fn += 1

tp = len([item for item in testing_set if item[3] == 1])

f1 = tp / tp + ((1/2) * (fp + fn))
'''

if load_model:
    model.load_state_dict(torch.load(model_name))
    model.train()

epoch_losses = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs}')
    epoch_loss = 0

    for i, (input, target) in enumerate(train_iterator):
        optimiser.zero_grad()
        yhat = model.forward(input).type(torch.FloatTensor).to(device)

        target = torch.tensor(target, dtype=torch.int64)
        #target = target.type(torch.IntTensor)
        target = target.to(device)
        print(yhat.size())
        print(target.size())
        loss = loss_fn(yhat.squeeze(0), target)
        print(loss)
        loss.requires_grad = True
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()

    torch.save(model.state_dict(), "./model_test.pt")
    print(epoch_loss / len(train_iterator))
    epoch_losses.append(epoch_loss / len(train_iterator))

print(epoch_losses)
plt.plot(epoch_losses)
plt.show()
