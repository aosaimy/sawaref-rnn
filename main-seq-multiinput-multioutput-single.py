# -*- coding: utf-8 -*-
'''
    An implementation of sequence to sequence learning
    for performing ensemble morphosyntactic analyses
'''
from __future__ import print_function
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
from prepare_data import SawarefData, padIndexes
from vis import SawarefVis
from character_table import colors, CharacterTable, eprint
import pandas as pd
import itertools


MYPATH = "/morpho/output/"
# Parameters for the model and dataset.
TRAINING_SIZE = 50000
EPOCHS = 3
# DIGITS = 3
# REVERSE = True
# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
EMBEDDINGS = 100
ITERATIONS = 10
REVERSE = False

sawarefData = SawarefData(MYPATH, EMBEDDINGS,feat_x=["MXpos","STpos","AMpos","FApos","STaspect","AMaspect"], feat_y=["QAsawalaha"])
pd.set_option('display.max_colwidth', 1000)
source = list(itertools.chain(*sawarefData.quran_sent))
df = pd.DataFrame(source,
                  columns=["sid", "aid", "wid", "mid"] +
                  sawarefData.features_map_x +
                  ["embeddings"] +
                  sawarefData.features_map_y)

def flattencolumns(df1, cols):
    df = pd.concat([pd.DataFrame(df1[x].values.tolist()).add_prefix(x) for x in cols], axis=1)
    return pd.concat([df, df1.drop(cols, axis=1)], axis=1)

def truncate(x):
    return x[:EMBEDDINGS]

df["embeddings"] = df["embeddings"].apply(truncate)
df = flattencolumns(df,["embeddings"])
df.set_index(["sid", "aid", "wid", "mid"], inplace=True)
df.sort_index(inplace=True)
SENTLEN = max(df.index.get_level_values("mid"))
df = df.reindex(padIndexes(df, max(df.index.get_level_values("mid"))), fill_value=0).sort_index()
dumm = pd.get_dummies(df, columns=sawarefData.features_map_x +
                  sawarefData.features_map_y)  #.reset_index().set_index("mid")  #.drop(["sid", "aid", "wid"], 1)
print("Done")

# dumm = dumm.reindex(padIndexes(dumm, max(df.index.get_level_values("mid"))), fill_value=0.0).sort_index()
# x_columns = [k + "_" + xx for k, x in sawarefData.features_set_x.items()
#              for xx in x]
# y_columns = [k + "_" + xx for k, x in sawarefData.features_set_y.items()
#              for xx in x]
x_columns = [y for f in sawarefData.features_map_x
             for y in dumm.columns if y.replace(f, "")[0] == "_"] + ["embeddings" + str(i) for i in range(EMBEDDINGS)]
y_columns = [y for f in sawarefData.features_map_y
             for y in dumm.columns if y.replace(f, "")[0] == "_"]
# remove indexing, index by mid only (sequence of morphemes)
questions = dumm.loc[:, x_columns]
expected = dumm.loc[:, y_columns]
# print(questions)
# exit()
if (questions.shape[0]%SENTLEN) != 0 or expected.shape[0] % SENTLEN != 0:
    eprint("Error: padding did not work correctly")
    eprint("Error: padding did not work correctly")
    print("questions shape = ", questions.shape, "= ",questions.shape[0]/SENTLEN)
    print("expected shape = ", expected.shape, "= ",expected.shape[0]/SENTLEN)
    print(expected)
    exit()

x = questions.values.reshape((int(questions.shape[0]/SENTLEN), SENTLEN, questions.shape[1]))
y = expected.values.reshape((int(expected.shape[0]/SENTLEN), SENTLEN, expected.shape[1]))
# print(y)
# exit()
# exit()
# ctable_x = CharacterTable(
#     set("-").union(set([xx for x in questions for xx in x])))

# ctable_y = CharacterTable(
#     set("-").union(set([xx for x in expected for xx in x])))

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
# MAXLEN = DIGITS + 1 + DIGITS


print('Total ayat questions:', len(questions))

print('Vectorization...')
# x = np.zeros((len(questions), SENTLEN,
#               len(ctable_x.chars)), dtype=np.bool)
# # len(ctable_x.chars) + EMBEDDINGS), dtype=np.bool)
# y = np.zeros((len(expected), SENTLEN,
#               len(ctable_y.chars)), dtype=np.bool)
# for i, sentence in enumerate(questions):
#     x[i] = ctable_x.encode(sentence, SENTLEN)
#     # x[i] = np.concatenate((ctable_x.encode([sentence], SENTLEN),
#     # np.array([embeddings[i]])), 1)
# for i, sentence in enumerate(expected):
#     y[i] = ctable_y.encode(sentence, SENTLEN)
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
    input_shape=(x.shape[1],x.shape[2])))
# input_shape=(None, len(ctable_x.chars) + EMBEDDINGS)))
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
model.add(layers.TimeDistributed(
    layers.Dense(y.shape[2])))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'sparse_categorical_accuracy'])
model.summary()


def pretty_join(arr):
    return "/".join(["-" if i[-2:]=="_0" else i for i in arr])
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
        q = pd.DataFrame(rowx[0], columns=questions.columns).drop(["embeddings" + str(i) for i in range(EMBEDDINGS)],axis=1).idxmax(axis=1)
        correct = pd.DataFrame(rowy[0], columns=expected.columns).idxmax(axis=1)
        # print(preds)
        # preds[0] = preds[0].argmax(axis=-1)
        # print(preds)
        res = np.zeros((y.shape[1],y.shape[2]))
        for i, c in enumerate(preds[0]):
            res[i, c] = 1
        guess = pd.DataFrame(res, columns=expected.columns).idxmax(axis=1)
        print('Q', pretty_join(q.values), end=' ')
        print('T', pretty_join(correct), end=' ')
        if (correct == guess).all():
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(pretty_join(guess))


# y_pred = []
# y_actual = []
# for i in range(len(y_val)):
#     y_actual.append(ctable_y.decode(y_val[i]))
#     y_pred.append(ctable_y.decode(
#         model.predict_classes(x_val[np.array([i])])[0],
#         calc_argmax=False))
    # print(y_actual[i], y_pred[i])

# SawarefVis(y_actual, y_pred,
#            ctable_y.chars)
