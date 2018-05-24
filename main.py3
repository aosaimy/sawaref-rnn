# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)
Input may optionally be reversed, shown to increase performance
in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies
between source and target.
Two digits reversed:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test
accuracy in 55 epochs
Three digits reversed:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test
accuracy in 100 epochs
Four digits reversed:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test
accuracy in 20 epochs
Five digits reversed:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test
accuracy in 30 epochs
'''
from __future__ import print_function
import matplotlib.pyplot as plt
import mplcursors
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
import json
from os import listdir
from os.path import join, isdir, exists
from character_table import CharacterTable, eprint
from prepare_data import SawarefData
import seaborn as sn
import pandas as pd


MYPATH = "/morpho/output/"
# Parameters for the model and dataset.
TRAINING_SIZE = 50000
EPOCHS = 30
# DIGITS = 3
# REVERSE = True
# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
SENTLEN = 1
EMBEDDINGS = 100
ITERATIONS = 1

sawarefData = SawarefData(MYPATH, EMBEDDINGS)

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
# MAXLEN = DIGITS + 1 + DIGITS


# Preparing the big training JSON object
quran = [f for f in listdir(MYPATH)
         if isdir(join(MYPATH, f)) if f[:1] == "q"
         if exists(join(MYPATH, f, "ALIGNED.json"))]
quran_sent = []
for j in quran:
    try:

        js = json.load(open(join(MYPATH, j, "ALIGNED.json")))
        quran_sent.append(js)
    except Exception as e:
        print(e)
        print(j)
        raise e
# print(quran_sent)
print(len(quran_sent))

# extract features
features_map_x = ["MXpos",
                  "STpos",
                  "AMpos",
                  "FApos",
                  "STaspect"
                  "AMaspect"
                  ]
features_map_y = ["QApos"]

features_set_x = dict()
for f in features_map_x:
    features_set_x[f] = set()

features_set_y = dict()
for f in features_map_y:
    features_set_y[f] = set()

questions = []
expected = []
embeddings = []
seen = set()
print('Vectorization...')

for ayah in quran_sent:
    # ayah_emb, words_x, words_y = [], [], []
    for w in ayah:
        q = set()
        for f in features_map_y:
            w[f] = (f + "na" if (w.get(f, "na") == "-----" or
                                 w.get(f, "na") == "-")
                    else f + w.get(f, "na"))
            if w[f] != f + "na":
                q.add(w[f])
                features_set_y[f].add(w[f])
        if len(q) == 0:
            continue
        expected.append("+".join(q))
        q.clear()
        for f in features_map_x:
            w[f] = (f + "na" if (w.get(f, "na") == "-----" or
                                 w.get(f, "na") == "-")
                    else f + w.get(f, "na"))
            if w[f] != f + "na":
                q.add(w[f])
                features_set_x[f].add(w[f])
        questions.append("+".join(q))
        q.clear()
        if len(w["embeddings"]) < EMBEDDINGS:
            eprint("EMBEDDINGS dimension not equal the provided ones",
                   len(w["embeddings"]), EMBEDDINGS)
            exit()
        elif len(w["embeddings"]) > EMBEDDINGS:
            w["embeddings"] = w["embeddings"][:EMBEDDINGS]

        embeddings.append(w["embeddings"])

# features_table = dict()
# for f in features_map_x:
#     features_table[f] = CharacterTable(features[f])

# the big features set
# bigset = set("-")
# for x in features_set_x:
#     if len(bigset.intersection(features_set_x[x])) > 0:
#         eprint("features are not unqie")
#         eprint(bigset.intersection(features_set_x[x]))
#         exit()
#     bigset = bigset.union(features_set_x[x])
# ctable_x = CharacterTable(bigset)
ctable_x = CharacterTable(set("-").union(set(questions)))
# print(max([len(x) for x in questions]))
# print(quran_sent[np.argmax(np.asarray([len(x) for x in questions]))])
# exit()
# bigset = set("-")
# for x in features_set_y:
#     if len(bigset.intersection(features_set_y[x])) > 0:
#         eprint("features are not unqie")
#         eprint(bigset.intersection(features_set_y[x]))
#         exit()
#     bigset = bigset.union(features_set_y[x])

# ctable_y = CharacterTable(bigset)
ctable_y = CharacterTable(set("-").union(set(expected)))

# print(ctable_x.chars)
# print(ctable_y.chars)
# print(ctable_x.chars)
# print(ctable_y.chars)
# print(ctable_x.encode(["aspectIMPV+MXposprep"], 1))
print(ctable_y.encode(["QAposV"], 1))
print(ctable_y.decode([0, 0], False))
# print(features_table["MXpos"].chars)
# print(sum(len(x) for x in quran_sent))
# exit()


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# def f():
#     return int(''.join(np.random.choice(list('0123456789'))
#                        for i in range(np.random.randint(1, DIGITS + 1))))


# while len(questions) < TRAINING_SIZE:
#     a, b = f(), f()
#     # Skip any addition questions we've already seen
#     # Also skip any such that x+Y == Y+x (hence the sorting).
#     key = tuple(sorted((a, b)))
#     if key in seen:
#         continue
#     seen.add(key)
#     # Pad the data with spaces such that it is always MAXLEN.
#     q = '{}+{}'.format(a, b)
#     query = q + ' ' * (MAXLEN - len(q))
#     ans = str(a + b)
#     # Answers can be of maximum size DIGITS + 1.
#     ans += ' ' * (DIGITS + 1 - len(ans))
#     if REVERSE:
#         # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
#         # space used for padding.)
#         query = query[::-1]
#     questions.append(query)
#     expected.append(ans)


print('Total ayat questions:', len(questions))

print('Vectorization...')
x = np.zeros((len(questions), SENTLEN,
              len(ctable_x.chars) + EMBEDDINGS), dtype=np.bool)
y = np.zeros((len(expected), SENTLEN, len(ctable_y.chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = np.concatenate((ctable_x.encode([sentence], SENTLEN),
                           np.array([embeddings[i]])), 1)
for i, sentence in enumerate(expected):
    y[i] = ctable_y.encode([sentence], SENTLEN)
# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(layers.Bidirectional(
    RNN(HIDDEN_SIZE),
    input_shape=(None, len(ctable_x.chars) + EMBEDDINGS)))
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.Dropout(0.5))
model.add(layers.RepeatVector(SENTLEN))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(ctable_y.chars))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'sparse_categorical_accuracy'])
model.summary()


# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, ITERATIONS + 1):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.

    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable_x.decode(rowx[0])
        correct = ctable_y.decode(rowy[0])
        guess = ctable_y.decode(preds[0], calc_argmax=False)
        # print('Q', q[::-1] if REVERSE else q, end=' ')
        print('Q', q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


y_pred = []
y_actual = []
for i in range(len(y_val)):
    y_actual.append(ctable_y.decode(y_val[i]))
    y_pred.append(ctable_y.decode(
        model.predict_classes(x_val[np.array([i])])[0],
        calc_argmax=False))
    # print(y_actual[i], y_pred[i])


# labels_indices = dict((c, i) for i, c in enumerate(labels))
# indices_labels = dict((i, c) for i, c in enumerate(labels))

# for i in range(len(y_val)):
#     y_pred[i] = labels_indices[y_pred[i]]
#     y_actual[i] = labels_indices[y_actual[i]]

# print(y_actual, y_pred)
cm = confusion_matrix(y_actual, y_pred, labels=ctable_y.chars)

df_cm = pd.DataFrame(cm, ctable_y.chars, ctable_y.chars)
print(df_cm)

# for label size
sn.set(font_scale=0.8)

# font size
sn.heatmap(df_cm, fmt="d", annot=True, robust=True, annot_kws={"size": 11},
           yticklabels=True, xticklabels=True, mask=df_cm == 0)
mplcursors.cursor(hover=True)
plt.show()
plt.savefig('confusion_matrix.png', format='png')

