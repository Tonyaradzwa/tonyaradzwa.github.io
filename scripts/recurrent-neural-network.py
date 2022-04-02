# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Preparing-data-for-use-as-NN-input" data-toc-modified-id="Preparing-data-for-use-as-NN-input-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Preparing data for use as NN input</a></span></li><li><span><a href="#Letting-the-NN-parameterize-words" data-toc-modified-id="Letting-the-NN-parameterize-words-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Letting the NN parameterize words</a></span></li><li><span><a href="#Adding-an-LSTM-layer" data-toc-modified-id="Adding-an-LSTM-layer-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Adding an LSTM layer</a></span></li><li><span><a href="#Classifiying-the-LSTM-output" data-toc-modified-id="Classifiying-the-LSTM-output-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Classifiying the LSTM output</a></span></li><li><span><a href="#Creating-training-and-validation-datasets" data-toc-modified-id="Creating-training-and-validation-datasets-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Creating training and validation datasets</a></span></li><li><span><a href="#Creating-the-Parts-of-Speech-LSTM-model" data-toc-modified-id="Creating-the-Parts-of-Speech-LSTM-model-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Creating the Parts of Speech LSTM model</a></span></li><li><span><a href="#Training" data-toc-modified-id="Training-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Training</a></span></li><li><span><a href="#Examining-results" data-toc-modified-id="Examining-results-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Examining results</a></span></li><li><span><a href="#Using-the-model-for-inference" data-toc-modified-id="Using-the-model-for-inference-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Using the model for inference</a></span></li></ul></div>

# %% [markdown]
# # Predicting parts of speech with an LSTM
#
# Let's preview the end result. We want to take a sentence and output the part of speech for each word in that sentence. Something like this:
#
# **Code**
#
# ```python
# new_sentence = "I is a teeth"
#
# ...
#
# predictions = model(processed_sentence)
#
# ...
# ```
#
# **Output**
#
# ```text
# I     => Noun
# is    => Verb
# a     => Determiner
# teeth => Noun
# ```

# %%
def ps(s):
    """Process String: convert a string into a list of lowercased words."""
    line = s.strip().replace(" ", "")
    return [c for c in line]


# %%
from pathlib import Path
import re

# read quesitons and answers from file
dataset_filename = Path("../train_data/arithmetic__mixed.txt")

# questions = [ ["1", "+" , "3"], ... ]
questions = []

# answers = [ [] ]
answers = []

with open(dataset_filename) as dataset_file:
    # Grabbing a subset of the entire file
    for i in range(100):
        line_q = dataset_file.readline().strip()
        line_a = dataset_file.readline().strip()

        questions.append([word.strip() for word in re.split(r'([+-/*()]|\s+)', line_q) if word.strip()])
        answers.append(eval(line_a))

# use zip to create dataset object
dataset = [(q,a) for q,a in zip(questions,answers)]

# %%
import torch

from fastprogress.fastprogress import progress_bar, master_bar

from random import shuffle

# %% [markdown]
# ## Preparing data for use as NN input
#
# We can't pass a list of plain text words and tags to a NN. We need to convert them to a more appropriate format.
#
# We'll start by creating a unique index for each word and tag.

# %%
word_to_index = {}
total_words = 0

for question, _ in dataset:

    total_words += len(question)

    for word in question:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

# %%
print("       Vocabulary Indices")
print("-------------------------------")

for word in sorted(word_to_index):
    print(f"{word:>14} => {word_to_index[word]:>2}")

print("\nTotal number of words:", total_words)
print("Number of unique words:", len(word_to_index))


# %% [markdown]
# ## Letting the NN parameterize words
#
# Once we have a unique identifier for each word, it is useful to start our NN with an [embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) layer. This layer converts an index into a vector of values.
#
# You can think of each value as indicating something about the word. For example, maybe the first value indicates how much a word conveys happiness vs sadness. Of course, the NN can learn any attributes and it is not limited to thinks like happy/sad, masculine/feminine, etc.
#
# **Creating an embedding layer**. An embedding layer is created by telling it the size of the vocabulary (the number of words) and an embedding dimension (how many values to use to represent a word).
#
# **Embedding layer input and output**. An embedding layer takes an index and return a matrix.

# %%
def convert_to_index_tensor(words, mapping):
    indices = [mapping[w] for w in words]
    return torch.tensor(indices, dtype=torch.long)


# %%
vocab_size = len(word_to_index)
embed_dim = 6  # Hyperparameter
embed_layer = torch.nn.Embedding(vocab_size, embed_dim)

# %%
# i = torch.tensor([word_to_index["the"], word_to_index["dog"]])
indices = convert_to_index_tensor(ps("15 + (7 + -17)/12"), word_to_index)
embed_output = embed_layer(indices)
indices.shape, embed_output.shape, embed_output

# %% [markdown]
# ## Adding an LSTM layer
#
# The [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM) layer is in charge of processing embeddings such that the network can output the correct classification. Since this is a recurrent layer, it will take into account past words when it creates an output for the current word.
#
# **Creating an LSTM layer**. To create an LSTM you need to tell it the size of its input (the size of an embedding) and the size of its internal cell state.
#
# **LSTM layer input and output**. An LSTM takes an embedding (and optionally an initial hidden and cell state) and outputs a value for each word as well as the current hidden and cell state).
#
# If you read the linked LSTM documentation you will see that it requires input in this format: (seq_len, batch, input_size)
#
# As you can see above, our embedding layer outputs something that is (seq_len, input_size). So, we need to add a dimension in the middle.

# %%
hidden_dim = 10  # Hyperparameter
num_layers = 5  # Hyperparameter
lstm_layer = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)

# %%
# The LSTM layer expects the input to be in the shape (L, N, E)
#   L is the length of the sequence
#   N is the batch size (we'll stick with 1 here)
#   E is the size of the embedding
lstm_output, _ = lstm_layer(embed_output.unsqueeze(1))
lstm_output.shape

# %% [markdown]
# ## Classifiying the LSTM output
#
# We can now add a fully connected, [linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear) layer to our NN to learn the correct part of speech (classification).
#
# **Creating a linear layer**. We create a linear layer by specifying the shape of the input into the layer and the number of neurons in the linear layer.
#
# **Linear layer input and output**. The input is expected to be (input_size, output_size) and the output will be the output of each neuron.

# %%
linear_layer = torch.nn.Linear(hidden_dim, 1)

# %%
linear_output = linear_layer(lstm_output)
linear_output.shape, linear_output

# %% [markdown]
# # Training an LSTM model

# %%
# Hyperparameters
valid_percent = 0.2  # Training/validation split

embed_dim = 7  # Size of word embedding
hidden_dim = 8  # Size of LSTM internal state
num_layers = 5  # Number of LSTM layers

learning_rate = 0.1
num_epochs = 2

# %% [markdown]
# ## Creating training and validation datasets

# %%
N = len(dataset)
vocab_size = len(word_to_index)  # Number of unique input words

# Shuffle the data so that we can split the dataset randomly
shuffle(dataset)

split_point = int(N * valid_percent)
valid_dataset = dataset[:split_point]
train_dataset = dataset[split_point:]

len(valid_dataset), len(train_dataset)


# %% [markdown]
# ## Creating the Parts of Speech LSTM model

# %%
class POS_LSTM(torch.nn.Module):
    """Part of Speach LSTM model."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, X):
        X = self.embed(X)
        X, _ = self.lstm(X.unsqueeze(1))
        return self.linear(X)


# %% [markdown]
# ## Training

# %%
def compute_accuracy(dataset):
    """A helper function for computing accuracy on the given dataset."""
    total_words = 0
    total_correct = 0

    model.eval()

    with torch.no_grad():
        for sentence, tags in dataset:
            sentence_indices = convert_to_index_tensor(sentence, word_to_index)
            tag_scores = model(sentence_indices).squeeze()
            predictions = tag_scores.argmax(dim=1)
            total_words += len(sentence)
            total_correct += sum(t == tag_list[p] for t, p in zip(tags, predictions))

    return total_correct / total_words


# %%
model = POS_LSTM(vocab_size, embed_dim, hidden_dim, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

mb = master_bar(range(num_epochs))

# accuracy = compute_accuracy(valid_dataset)
# print(f"Validation accuracy before training : {accuracy * 100:.2f}%")

for epoch in mb:

    # Shuffle the data for each epoch (stochastic gradient descent)
    shuffle(train_dataset)

    model.train()

    for sentence, tags in progress_bar(train_dataset, parent=mb):
        model.zero_grad()
        
        sentence = convert_to_index_tensor(sentence, word_to_index)
#         tags = convert_to_index_tensor(tags, tag_to_index)

        tag_scores = model(sentence)

        break
#         loss = criterion(tag_scores.squeeze(), tags)

#         loss.backward()
#         optimizer.step()

# accuracy = compute_accuracy(valid_dataset)
# print(f"Validation accuracy after training : {accuracy * 100:.2f}%")

# %% [markdown]
# ## Examining results
#
# Here we look at all words that are misclassified by the model

# %%
print("\nMis-predictions after training on entire dataset")
header = "Word".center(14) + " | True Tag | Prediction"
print(header)
print("-" * len(header))

with torch.no_grad():
    for sentence, tags in dataset:
        sentence_indices = convert_to_index_tensor(sentence, word_to_index)
        tag_scores = model(sentence_indices)
        predictions = tag_scores.squeeze().argmax(dim=1)
        for word, tag, pred in zip(sentence, tags, predictions):
            if tag != tag_list[pred]:
                print(f"{word:>14} |     {tag}    |    {tag_list[pred]}")

# %% [markdown]
# ## Using the model for inference

# %%
new_sentence = "3 + 3"

# Convert sentence to lowercase words
sentence = ps(new_sentence)

# Check that each word is in our vocabulary
for word in sentence:
    assert word in word_to_index

# Convert input to a tensor
sentence = convert_to_index_tensor(sentence, word_to_index)

# Compute prediction
predictions = model(sentence)
predictions = predictions.squeeze().argmax(dim=1)

# Print results
for word, tag in zip(ps(new_sentence), predictions):
    print(word, "=>", tag_list[tag.item()])

# %% [markdown]
# Things to try:
#
# - compare with fully connected network
# - compare with CNN
# - compare with transformer

# %%

# %%

# %%
