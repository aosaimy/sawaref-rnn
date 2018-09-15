# -*- coding: utf-8 -*-
'''
    An implementation of sequence to sequence learning
    for performing ensemble morphosyntactic analyses
'''
from __future__ import print_function
# from keras.preprocessing.sequence import pad_sequences
import numpy as np
from six.moves import range
from prepare_data import SawarefData, padIndexes
from character_table import colors, CharacterTable, eprint
import pandas as pd
import itertools
import re
import pickle
import sys

# do not import in interactive mode
from vis import SawarefVis
from keras.models import Sequential, Model, load_model
from keras import layers
from keras.utils import plot_model


MYPATH = "/morpho/output/"
# Parameters for the model and dataset.
TRAINING_SIZE = 50000
EPOCHS = 1
EMBEDDINGS = 100
# DIGITS = 3
# REVERSE = True
# Try replacing GRU, or SimpleRNN.
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
ITERATIONS = 10
REVERSE = False
MODEL_NAME = "main-seq-multiinput-multioutput-segmentation.keras"
DATA_PICKLE = "main-seq-multiinput-multioutput-segmentation.pickle"
RNN = layers.LSTM

feat_x=["MXpos", "STpos", "AMpos", "FApos", 
        # "STaspect", "AMaspect", "MXaspect", "FAaspect",
        # "STperson", "AMperson", "MXperson", "FAperson",
        # "STgender", "AMgender", "MXgender", "FAgender",
        # "STnumber", "AMnumber", "MXnumber", "FAnumber",
        # "STcase", "AMcase", "MXcase", "FAcase",
        # "STvoice", "AMvoice", "MXvoice", "FAvoice",
        # "STmood", "AMmood", "MXmood", "FAmood",
        # "STstate", "AMstate", "MXstate", "FAstate"
        ]
feat_y=["QApos",] #"QAaspect", "QAperson", "QAgender", "QAnumber", "QAcase", "QAvoice", "QAmood", "QAstate"])
strings_x=["word"]
strings_y=["QAutf8"]
pd.set_option('display.max_colwidth', 1000)


def pretty_join(arr):
    # return "/".join(["-" if i[-2:] == "_0" else i for i in arr])
    return "/".join(['+'.join([pretty_value(x[1]) for x in arr.columns[row == 1]])
                     for index, row in arr.iterrows()])


def pretty_value(colum_value):
    return "-" if colum_value[-2:] == "_0" else re.sub(".*_","",colum_value)


def getValuesAndReshape(df, middle_dim):
    return df.values.reshape((df.shape[0]//middle_dim, middle_dim, df.shape[1]))


def flattencolumns(df1, cols):
    df = pd.concat([pd.DataFrame(df1[x].values.tolist()).add_prefix(x)
                    for x in cols], axis=1)
    return pd.concat([df, df1.drop(cols, axis=1)], axis=1)


def truncate(x):
    return x[:EMBEDDINGS]


def removeDiac(x):
    return re.sub('[ًٌٍَُِّْ]', '', x)


def joinMorphemesStrings(arr):
    return "+".join([x for x in arr if isinstance(x, float) == False and len(x) != 0 and x != "-----" and x != "-"])

argv = sys.argv

if "load" not in argv:
    ## 1. Load the data from Sawaref repo, feat_x and feat_y are the cateogriacal features
    sawarefData = SawarefData(MYPATH, EMBEDDINGS,
                              feat_x=["MXpos", "STpos", "AMpos", "FApos", 
                                      # "STaspect", "AMaspect", "MXaspect", "FAaspect",
                                      # "STperson", "AMperson", "MXperson", "FAperson",
                                      # "STgender", "AMgender", "MXgender", "FAgender",
                                      # "STnumber", "AMnumber", "MXnumber", "FAnumber",
                                      # "STcase", "AMcase", "MXcase", "FAcase",
                                      # "STvoice", "AMvoice", "MXvoice", "FAvoice",
                                      # "STmood", "AMmood", "MXmood", "FAmood",
                                      # "STstate", "AMstate", "MXstate", "FAstate"

                                      ],
                              strings_x=["word"],
                              strings_y=["QAutf8"],
                              feat_y=["QApos",]) #"QAaspect", "QAperson", "QAgender", "QAnumber", "QAcase", "QAvoice", "QAmood", "QAstate"])
    source = list(itertools.chain(*sawarefData.quran_sent))
    df = pd.DataFrame(source,
                      columns=["sid", "aid", "wid", "mid"] +
                      feat_x +
                      ["embeddings","word","QAutf8"] +
                      feat_y)
    df["embeddings"] = df["embeddings"].apply(truncate)
    df = flattencolumns(df, ["embeddings"])
    df.set_index(["sid", "aid", "wid", "mid"], inplace=True)
    df.sort_index(inplace=True)

    ## 2. Pad the rows according to the longest word (in # of morphemes)
    SENTLEN = max(df.index.get_level_values("mid"))
    df = df.reindex(padIndexes(
        df, max(df.index.get_level_values("mid"))), fill_value=0).sort_index()

    ## 3. Get the hot encoding of all caterogirical data (see columns attr)
    dumm = pd.get_dummies(df, columns=feat_x +
                          feat_y)

    ## 4. Add two-level columns for easy indexing later (wid, mid)
    EXAMPLES_LEN = df.shape[0]//SENTLEN
    new_columns = []
    for x in dumm.columns:
        new_columns.append(re.sub('(_.*|[0-9]*)', '', x))
    dumm.columns = [new_columns, dumm.columns]
    dumm.index = [[x for x in range(EXAMPLES_LEN) for _ in range(SENTLEN)],
                  [x for _ in range(EXAMPLES_LEN) for x in range(SENTLEN)]]
    dumm = dumm.sort_index(axis=1)

    ## 5. Prepare string columns: 
    # a. clean all padded rows
    strings = strings_x + strings_y
    for s in strings:
      df.loc[df[s]==0,s] = ""

    df["word_undiac"] = df["word"].apply(removeDiac)

    # b. group them by morpheme and join with "+"" 
    df_strings = pd.DataFrame({x:df[x].groupby(level=[0,1,2]).apply(joinMorphemesStrings) for x in strings})

    # c. encode them in one hot encoding
    charset = set("+").union(*[list(set("".join(df_strings[x]+"-"))) for x in strings])
    STRING_LENGTH = max([len(x) for k in strings for x in df_strings[k]])
    ctable = CharacterTable(charset,STRING_LENGTH)
    ### Now we have one shape for all strings: (STRING_LENGTH, len(charset))
    for x in strings:
      df_strings[x] = df_strings[x].apply(ctable.encode)
    values = {x:np.stack(df_strings[x].values) for x in strings}
    if "save" in argv:
        pickle.dump([dumm,values], open(DATA_PICKLE, mode="wb"))
else:
    arr = pickle.load(
            open(DATA_PICKLE, mode="rb"), encoding="UTF8")
    (dumm,values) = arr[0], arr[1]
    SENTLEN = max(dumm.index.get_level_values(1))+1
    EXAMPLES_LEN = dumm.shape[0]//SENTLEN

# 6. Shuffle (x, y) in unison
indices = list(range(EXAMPLES_LEN))
np.random.shuffle(indices)

# 7. Explicitly set apart 10% for validation data that we never train over.
split_at = EXAMPLES_LEN - EXAMPLES_LEN // 10

values = {x:(values[x][:split_at],values[x][split_at:]) for x in values}
x_train = dumm.loc[(indices[:split_at], slice(None)),(feat_x, slice(None))]
x_val = dumm.loc[(indices[split_at:], slice(None)),(feat_x, slice(None))]
# (y_train, y_val) = expected.loc[(indices[:split_at], slice(None)),:], expected.loc[(indices[split_at:], slice(None)),:]
y_trains =  {i: dumm.loc[(indices[:split_at], slice(None)), (i, slice(None))] for i in feat_y}
# y_trains[i] = y_train.loc[:, (i, slice(None))]
# y_vals[i] = y_val.loc[:, (i, slice(None))]
y_vals = {i: dumm.loc[(indices[split_at:], slice(None)), (i, slice(None))] for i in feat_y}

outputs_objects = {i:getValuesAndReshape(y_trains[i], SENTLEN) for i in y_trains}
outputs_objects["strings_output"] = values["QAutf8"][0]
outputs_objects_val = {i:getValuesAndReshape(y_vals[i], SENTLEN) for i in y_vals}
outputs_objects_val["strings_output"] = values["QAutf8"][1]
data = {'input': {'main_input': getValuesAndReshape(x_train, SENTLEN),'strings_input': values["word"][0]},
        'output': outputs_objects, 
        'val': ({'main_input': getValuesAndReshape(x_val, SENTLEN),'strings_input': values["word"][1]},
                            outputs_objects_val)}

# 8. Some info
print('Training Data:')
print(x_train.shape)
print([y_trains[i].shape for i in feat_y])
print('Validation Data:')
print(x_val.shape)
print([y_vals[i].shape for i in feat_y])


if "train" in argv:

    print('Build model...')
    # model = Sequential()

    # For strings
    strings_input = layers.Input(shape=(STRING_LENGTH, len(charset)), name='strings_input')
    lstm_strings_encoder = layers.Bidirectional(RNN(HIDDEN_SIZE))(strings_input)
    repeat_strings_out = layers.RepeatVector(STRING_LENGTH)(lstm_strings_encoder)
    rnn_out = RNN(HIDDEN_SIZE, return_sequences=True)(repeat_strings_out)
    strings_output = layers.TimeDistributed(
          layers.Dense(
            len(charset), 
            activation="softmax"), name="strings_output")(rnn_out)


    # For categoricals
    main_input = layers.Input(shape=(SENTLEN, x_train.shape[1]), name='main_input')

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
    for i in feat_y:
        outputs.append(
          layers.TimeDistributed(
          layers.Dense(
            y_trains[i].shape[1], 
            activation="softmax"), name=i)(rnn_out))
        # time_out = layers.TimeDistributed(layers.Dense(y_trains[i].shape[1]))(rnn_out)
        # outputs.append(layers.Activation('softmax', name=i)(time_out))
    outputs.append(strings_output)

    model = Model(inputs=[main_input, strings_input], outputs=outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'sparse_categorical_accuracy'])

if "test" in argv:
    model = load_model(MODEL_NAME)

model.summary()
plot_model(model, to_file='model.png', show_shapes=True)


    # Train the model each generation and show predictions against the validation
    # dataset.

# print(outputs_objects)
for iteration in range(1, ITERATIONS + 1):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    if "train" in argv:
        history = model.fit(data['input'], data['output'],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=data['val'])
        model.save(MODEL_NAME)
    # Select 10 samples from the validation set at random so we can visualize
    # errors.

    for _ in range(10):
        ind = np.random.randint(0, len(x_val.index))
        ind = (x_val.index[ind][0],slice(None))
        string_val = values["word"][1][ind:ind+1]
        print(string_val)
        rowx = x_val.loc[ind]

        preds = model.predict([getValuesAndReshape(rowx, SENTLEN), string_val], verbose=0)
        preds = [np.argmax(x,axis=-1) for x in preds]
        print(preds)

        rowy = dict()
        print('Q', pretty_join(rowx))
        for i, v in enumerate(feat_y):
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
