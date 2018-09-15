# -*- coding: utf-8 -*-
'''
    An implementation of sequence to sequence learning
    for performing ensemble morphosyntactic analyses
'''
from __future__ import print_function
# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras import layers
import numpy as np
from six.moves import range
from prepare_data import SawarefData, padIndexes
from vis import SawarefVis
from character_table import colors, eprint
import pandas as pd
import itertools
import re


MYPATH = "/morpho/output/"
# Parameters for the model and dataset.
TRAINING_SIZE = 50000
EPOCHS = 1
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


def pretty_join(arr):
    # return "/".join(["-" if i[-2:] == "_0" else i for i in arr])
    return "/".join(['+'.join([pretty_value(x[1]) for x in arr.columns[row == 1]])
                     for index, row in arr.iterrows()])


def pretty_value(colum_value):
    return "-" if colum_value[-2:] == "_0" else re.sub(".*_","",colum_value)


def getValuesAndReshape(df, middle_dim):
    if(len(df.shape)==1):
        return df.values#.reshape((df.shape[0]//middle_dim, middle_dim))
    else:
        return df.values.reshape((df.shape[0]//middle_dim, middle_dim, df.shape[1]))


def flattencolumns(df1, cols):
    df = pd.concat([pd.DataFrame(df1[x].values.tolist()).add_prefix(x)
                    for x in cols], axis=1)
    return pd.concat([df, df1.drop(cols, axis=1)], axis=1)


def truncate(x):
    return x[:EMBEDDINGS]


sawarefData = SawarefData(MYPATH, EMBEDDINGS,
                          feat_x=["MXpos", "STpos", "AMpos", "FApos", 
                                  "STaspect", "AMaspect", "MXaspect", "FAaspect",
                                  "STperson", "AMperson", "MXperson", "FAperson",
                                  "STgender", "AMgender", "MXgender", "FAgender",
                                  "STnumber", "AMnumber", "MXnumber", "FAnumber",
                                  "STcase", "AMcase", "MXcase", "FAcase",
                                  "STvoice", "AMvoice", "MXvoice", "FAvoice",
                                  "STmood", "AMmood", "MXmood", "FAmood",
                                  "STstate", "AMstate", "MXstate", "FAstate",
                                  ],
                          feat_y=["QApos", "QAaspect", "QAperson", "QAgender", "QAnumber", "QAcase", "QAvoice", "QAmood", "QAstate"])
pd.set_option('display.max_colwidth', 1000)
source = list(itertools.chain(*sawarefData.quran_sent))
df = pd.DataFrame(source,
                  columns=["sid", "aid", "wid", "mid"] +
                  sawarefData.features_map_x +
                  ["embeddings"] +
                  sawarefData.features_map_y)

df["embeddings"] = df["embeddings"].apply(truncate)
df = flattencolumns(df, ["embeddings"])
df.set_index(["sid", "aid", "wid", "mid"], inplace=True)
df.sort_index(inplace=True)
SENTLEN = max(df.index.get_level_values("mid"))
df = df.reindex(padIndexes(
    df, max(df.index.get_level_values("mid"))), fill_value=0).sort_index()
EXAMPLES_LEN = df.shape[0]//SENTLEN
dumm = pd.get_dummies(df, columns=sawarefData.features_map_x +
                      sawarefData.features_map_y)

## add two-level columns for easy indexing later

new_columns = []
for x in dumm.columns:
    new_columns.append(re.sub('(_.*|[0-9]*)', '', x))

dumm.columns = [new_columns, dumm.columns]
dumm.index = [[x for x in range(EXAMPLES_LEN) for _ in range(SENTLEN)],
              [x for _ in range(EXAMPLES_LEN) for x in range(SENTLEN)]]
dumm = dumm.sort_index(axis=1)

## add index 

# print(dumm)
print("Done")

# dumm = dumm.reindex(padIndexes(dumm, max(df.index.get_level_values(
# "mid"))), fill_value=0.0).sort_index()
# x_columns = [k + "_" + xx for k, x in sawarefData.features_set_x.items()
#              for xx in x]
# y_columns = [k + "_" + xx for k, x in sawarefData.features_set_y.items()
#              for xx in x]
# x_columns = ([y for f in sawarefData.features_map_x
#               for y in dumm.columns if y.replace(f, "")[0] == "_"] +
#              ["embeddings" + str(i) for i in range(EMBEDDINGS)])
# y_columns = [y for f in sawarefData.features_map_y
#              for y in dumm.columns if y.replace(f, "")[0] == "_"]
# remove indexing, index by mid only (sequence of morphemes)

questions = dumm.loc[:, (sawarefData.features_map_x, slice(None))]
# expected = dumm.loc[:, y_columns]

expected = dumm.loc[:, (sawarefData.features_map_y, slice(None))]

# print(questions)
# exit()
if (questions.shape[0] % SENTLEN) != 0 or expected.shape[0] % SENTLEN != 0:
    eprint("Error: padding did not work correctly")
    eprint("Error: padding did not work correctly")
    print("questions shape = ", questions.shape,
          "= ", questions.shape[0] / SENTLEN)
    print("expected shape = ", expected.shape,
          "= ", expected.shape[0] / SENTLEN)
    print(expected)
    exit()

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
# indices = np.arange(len(y))
indices = list(range(EXAMPLES_LEN))
np.random.shuffle(indices)
# x = x[indices]
# y = y[indices]
# questions_shuffled = questions.loc[(indices[:split_at], None),:]

# Explicitly set apart 10% for validation data that we never train over.
split_at = EXAMPLES_LEN - EXAMPLES_LEN // 10
(x_train, x_val) = questions.loc[(indices[:split_at], slice(None)),:], questions.loc[(indices[split_at:], slice(None)),:]

(y_train, y_val) = expected.loc[(indices[:split_at], slice(None)),:], expected.loc[(indices[split_at:], slice(None)),:]
y_trains = dict()
y_vals = dict()
for i in sawarefData.features_map_y:
    y_trains[i] = y_train.loc[:, (i, slice(None))]
    # y_trains[i] = getValuesAndReshape(y_trains[i],SENTLEN)
    y_vals[i] = y_val.loc[:, (i, slice(None))]
    # y_vals[i] = getValuesAndReshape(y_vals[i], SENTLEN)

print('Training Data:')
print(x_train.shape)
print([y_trains[i].shape for i in sawarefData.features_map_y])

print('Validation Data:')
print(x_val.shape)
print([y_vals[i].shape for i in sawarefData.features_map_y])

print('Build model...')
# model = Sequential()

main_input = layers.Input(shape=(SENTLEN, questions.shape[1]), name='main_input')


lstm_out = layers.Bidirectional(RNN(HIDDEN_SIZE))(main_input)
# input_shape=(None, len(ctable_x.chars) + EMBEDDINGS)))
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
dropout_out = layers.Dropout(0.5)(lstm_out)
repeat_out = layers.RepeatVector(SENTLEN)(dropout_out)
# The decoder RNN could be multiple layers stacked or a single layer.
rnn_out = RNN(HIDDEN_SIZE, return_sequences=True)(repeat_out)
for _ in range(LAYERS-1):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    rnn_out = RNN(HIDDEN_SIZE, return_sequences=True)(rnn_out)

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
outputs = []
for i in sawarefData.features_map_y:
    time_out = layers.TimeDistributed(layers.Dense(y_trains[i].shape[1]))(rnn_out)
    outputs.append(layers.Activation('softmax', name=i)(time_out))

model = Model(inputs=[main_input], outputs=outputs)

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
    history = model.fit(
        {'main_input': getValuesAndReshape(x_train, SENTLEN)},
        {i:getValuesAndReshape(y_trains[i], SENTLEN) for i in sawarefData.features_map_y},
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=({'main_input': getValuesAndReshape(x_val, SENTLEN)},
                            {i:getValuesAndReshape(y_vals[i], SENTLEN) for i in sawarefData.features_map_y}))

    model.save("mymodel.keras")
    # Select 10 samples from the validation set at random so we can visualize
    # errors.

    for i in range(10):
        ind = np.random.randint(0, len(x_val.index))
        ind = (x_val.index[ind][0],slice(None))
        rowx = x_val.loc[ind]

        preds = model.predict(getValuesAndReshape(rowx, SENTLEN), verbose=0)
        preds = [np.argmax(x,axis=-1) for x in preds]
        print(preds)

        rowy = dict()
        print('Q', pretty_join(rowx))
        for i, v in enumerate(sawarefData.features_map_y):
            rowy[v] = {"correct": y_vals[v].loc[ind]}

            res = np.zeros((SENTLEN,rowy[v]["correct"].shape[1]))
            for ii, c in enumerate(preds[i][0]):
                res[ii, c] = 1
            rowy[v]["pred"] =  pd.DataFrame(res, columns=y_vals[v].columns)



            print('T', pretty_join(rowy[v]["correct"]), end=' ')
            if (rowy[v]["correct"].values == rowy[v]["pred"].values).all():
                print(colors.ok + '☑' + colors.close, end=' ')
            else:
                print(colors.fail + '☒' + colors.close, end=' ')
            print(pretty_join(rowy[v]["pred"]))


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
