# coding: utf-8

# # 1. Import

'''
    An implementation of sequence to sequence learning
    for performing ensemble morphosyntactic analyses
'''
from __future__ import print_function
# from keras.preprocessing.sequence import pad_sequences
import numpy as np
from six.moves import range
from prepare_data import SawarefData, padIndexes
from character_table import CharacterTable  # , eprint
import pandas as pd
import itertools
import re
import pickle
# import sys
import datetime
from keras.callbacks import TensorBoard
from buckwalter import utf2bw  # , bw2utf
from pprint import pprint
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# do not import in interactive mode
# from vis import SawarefVis
from keras.models import Model, load_model
from keras import layers
from keras.callbacks import EarlyStopping, Callback
from keras.utils import plot_model

from gensim.models import FastText


# # 2. Constants

NAME = input("What is the name of this experiment? ")
print("\nNAME=", NAME)
thedate = datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M")


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        metrics = self.model.evaluate(x, y, verbose=0)
        for i, x in enumerate(self.model.metrics_names):
            logs["test_" + x] = metrics[i]
        logs['test_acc'] = np.mean([
            logs["test_" + x] for x in self.model.metrics_names
            if x[-4:] == '_acc'
        ])
        logs['val_acc'] = np.mean([
            logs["val_" + x] for x in self.model.metrics_names
            if x[-4:] == '_acc'
        ])
        logs['train_acc'] = np.mean(
            [logs[x] for x in self.model.metrics_names if x[-4:] == '_acc'])
        return logs


MYPATH = "/morpho/output/"
# Parameters for the model and dataset.
TRAINING_SIZE = 50000
EPOCHS = 100
EMBEDDINGS = 100
# DIGITS = 3
# REVERSE = True
# Try replacing GRU, or SimpleRNN.
HIDDEN_SIZE = 128
BATCH_SIZE = 64
LAYERS = 1
ITERATIONS = 10
REVERSE = False
MODEL_NAME = "main-seq-multiinput-multioutput-segmentation.keras"
DATA_PICKLE = "main-seq-multiinput-multioutput-segmentation.pickle"
RNN = layers.LSTM
CATS_EMBEDDING = 5
VAL_SPLIT = 1  # means 0.1 of data is for test
WORD_EMB = False

if WORD_EMB:
    print("Loading FastText model", flush=True)
    emb = FastText.load_fasttext_format(
        "/Users/abbander/Leeds/OpenArabic/data/classical.bin")
    print(" Loaded.")

# # 3. Config

# `feat_x` is the input categorical features
# `feat_y` is the output categorical features
# `strings_x` is the input character-based strings features
# `strings_y` is the output character-based strings features

feat_x = [
    "MXpos", "STpos", "AMpos", "FApos", "STaspect", "AMaspect", "MXaspect",
    "FAaspect", "STperson", "AMperson", "MXperson", "FAperson", "STgender",
    "AMgender", "MXgender", "FAgender", "STnumber", "AMnumber", "MXnumber",
    "FAnumber", "STcase", "AMcase", "MXcase", "FAcase", "STvoice",
    "AMvoice", "MXvoice", "FAvoice", "STmood", "AMmood", "MXmood",
    "FAmood", "STstate", "AMstate", "MXstate", "FAstate"
]
feat_x = []
print("Warning! feat_x is empty")
feat_y = [
    "QApos", "QAaspect", "QAperson", "QAgender", "QAnumber", "QAcase",
    "QAvoice", "QAmood", "QAstate"
]
strings_x = ["QAwutf8"]
strings_y = ["QAutf8"]


def pretty_join(arr):
    if isinstance(arr, pd.Series):
        arr = arr.to_frame().T
    if isinstance(arr.columns, pd.core.index.MultiIndex):
        return "/".join([
            '+'.join([
                x[1] for x in arr.columns[row == 1]
                if x[1][-2:] != "na" and x[1][-2:] != "_0"
            ]) for index, row in arr.iterrows()
        ])
    else:
        return "/".join([
            '+'.join([
                x for x in arr.columns[row == 1]
                if x[-2:] != "na" and x[-2:] != "_0"
            ]) for index, row in arr.iterrows()
        ])


def pretty_value(colum_value):
    return re.sub(".*_", "", colum_value)


def getValuesAndReshape(df, middle_dim):
    return df.values.reshape((df.shape[0] // middle_dim, middle_dim, -1))


def flattencolumns(df1, cols):
    df = pd.concat(
        [pd.DataFrame(df1[x].values.tolist()).add_prefix(x) for x in cols],
        axis=1)
    return pd.concat([df, df1.drop(cols, axis=1)], axis=1)


def truncate(x):
    return x[:EMBEDDINGS]


def removeDiac(x):
    return re.sub('[ًٌٍَُِّْ]', '', x.replace("ٱ", "ا"))


def padStringWithSpaces(x):
    return x + ' ' * (STRING_LENGTH - len(x))


def joinMorphemesStrings(arr):
    return "+".join([
        x for x in arr if (isinstance(x, float) is False and
                           len(x) != 0 and x != "-----" and x != "-")
    ])


def emb_encode(x):
    return np.zeros(emb.vector_size) if x.strip() == "" else emb[x]


def fullprint(*args, **kwargs):
    opt = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)


def getMorphemeBasedIndeicies(arr):
    return np.array([list(range(x * 5, x * 5 + 5)) for x in arr]).flatten()


def getData(cat_source,
            str_source,
            cats_feats=[],
            strs_feats=[],
            embeddings=False):
    data = {
        **{
            i: getValuesAndReshape(
                cat_source[i].idxmax(axis=1).apply(
                    lambda x: embeddingInputLists[i].index(x)),
                SENTLEN)
            # getValuesAndReshape(cat_source.loc[:, ("vals", i)], SENTLEN)
            if (i in embeddingInputSets and
                len(embeddingInputSets[i]) > CATS_EMBEDDING)
            else getValuesAndReshape(cat_source[i], SENTLEN)
            for i in cats_feats
        },
        **{
            i: np.stack(str_source[i + "_onehot"].values)
            for i in strs_feats
        },
        # "embeddings": getValuesAndReshape(cat_source[i], SENTLEN),
    }
    if embeddings:
        for i in strs_feats:
            data[i + "_emb"] = np.stack(str_source[i + "_emb"].values)
    return data


def getEmbedding(input):
    name = input.name.split("_")[0]
    if name in feat_x and len(embeddingInputSets[name]) > CATS_EMBEDDING:
        return layers.Reshape((SENTLEN, -1))(
            layers.Embedding(len(embeddingInputSets[name]), 2,
                             input_length=SENTLEN)(input))
    else:
        return layers.Dropout(0.1)(input)

# # 4. Loading data from sawaref


sawarefData = SawarefData(
    MYPATH,
    EMBEDDINGS,
    feat_x=feat_x,
    strings_x=strings_x,
    strings_y=strings_y,
    feat_y=feat_y)

source = list(itertools.chain(*sawarefData.quran_sent))
df = pd.DataFrame(
    source,
    columns=["sid", "aid", "wid", "mid"] + feat_x + strings_x + strings_y +
    ["embeddings"] + feat_y)
df["embeddings"] = df["embeddings"].apply(truncate)
df = flattencolumns(df, ["embeddings"])
df.set_index(["sid", "aid", "wid", "mid"], inplace=True)
df.sort_index(inplace=True)

# 2. Pad the rows according to the longest word (in # of morphemes)
SENTLEN = max(df.index.get_level_values("mid"))
df = df.reindex(
    padIndexes(df, max(df.index.get_level_values("mid"))),
    fill_value=0).sort_index()

# 3. Get the hot encoding of all caterogirical data (see columns attr)
dumm = pd.get_dummies(df, columns=feat_x + feat_y)

# 4. Add two-level columns for easy indexing later (wid, mid)
EXAMPLES_LEN = df.shape[0] // SENTLEN
new_columns = []
for x in dumm.columns:
    new_columns.append(re.sub('(_.*|[0-9]*)', '', x))
dumm.columns = [new_columns, dumm.columns]
dumm.index = [[x for x in range(EXAMPLES_LEN) for _ in range(SENTLEN)],
              [x for _ in range(EXAMPLES_LEN) for x in range(SENTLEN)]]
dumm.sort_index(axis=1, inplace=True)

# 5. Find possible values of each cat


def getSet(df):
    results = set()
    df.apply(results.add)
    return results


embeddingInputSets = {i: getSet(df[i]) for i in feat_x}

# embeddingInputSets

feat_x = list(
    set(feat_x) -
    set([x for x in feat_x if len(embeddingInputSets[x]) <= 1]))

df2 = pd.concat([df.reset_index(), dumm.reset_index()], axis=1)
# df2.set_index(["sid", "aid", "wid", "mid"], inplace=True)
df2.index = [[x for x in range(EXAMPLES_LEN) for _ in range(SENTLEN)],
             [x for _ in range(EXAMPLES_LEN) for x in range(SENTLEN)]]

df2.drop(['embeddings' + str(x) for x in range(100)], inplace=True, axis=1)
# df2.columns = [(x,"val") if isinstance(x,str)
# else x  for x in df2.columns]

df2.index = [[x for x in range(EXAMPLES_LEN) for _ in range(SENTLEN)],
             [x for _ in range(EXAMPLES_LEN) for x in range(SENTLEN)]]

df2.columns = [[
    "vals" if isinstance(x, str) else x[0] for x in df2.columns
], [x if isinstance(x, str) else x[1] for x in df2.columns]]

df2.sort_index(axis=1, inplace=True)

del df
del dumm

# 5. Prepare string columns:
# a. clean all padded rows
strings = strings_x + strings_y
for s in strings:
    #         print(df2[s])
    df2.loc[df2[("vals", s)] == 0, ("vals", s)] = ""
    df2.loc[pd.isna(df2[("vals", s)]), ("vals", s)] = ""
for x in strings:
    df2[(x, "undiac")] = df2[("vals", x)].apply(removeDiac)

df2["vals", "QAutf8"].groupby(level=[0]).apply(joinMorphemesStrings).head()

# b. group them by morpheme and join with "+""
df_strings = pd.DataFrame({
    x: df2["vals", x].groupby(level=[0]).apply(joinMorphemesStrings)
    for x in strings
})

# df_strings

# c. pad joined morphemes
STRING_LENGTH = max([len(x) for k in strings for x in df_strings[k]])
for s in strings:
    df_strings[s] = df_strings[s].apply(padStringWithSpaces)

# d. encode them in one hot encoding
charset = set("+").union(
    *[list(set("".join(df_strings[x] + "-"))) for x in strings])
ctable = CharacterTable(charset, STRING_LENGTH)
# Now we have one shape for all strings: (STRING_LENGTH, len(charset))
for x in strings:
    df_strings[x + "_onehot"] = df_strings[x].apply(ctable.encode)
df_strings['num'] = [x for x in range(len(df_strings))]
df_strings.set_index('num', append=True, inplace=True)

if WORD_EMB:
    # e. remove diac
    for x in strings:
        df_strings[x + "_undiac"] = df_strings[x].apply(removeDiac)

    # f. encode them as dense vector using fastText
    for x in strings:
        df_strings[x + "_emb"] = df_strings[x + "_undiac"].apply(emb_encode)


print("START FOR LOOP ")

np.random.seed(0)
TEST_SPLIT = 1  # means 0.1 of data is for test
print("TEST_SPLIT=", TEST_SPLIT)
thedate = (str(TEST_SPLIT) +
           "+" + datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M"))
# df_strings[:][0:10]

# # 5. Save (or load) the data

# pickle.dump([df2, df_strings], open(DATA_PICKLE, mode="wb"))

# arr = pickle.load(open(DATA_PICKLE, mode="rb"), encoding="UTF8")
# (dumm, df_strings) = arr[0], arr[1]
# SENTLEN = max(dumm.index.get_level_values(1)) + 1
# EXAMPLES_LEN = dumm.shape[0] // SENTLEN

# # 6. Prepare splits

# 6. Shuffle (x, y) in unison
indices = list(range(EXAMPLES_LEN))
np.random.shuffle(indices)

# indices

# 7. Explicitly set apart 10% for validation data that we never train over.
test_split_at = int(EXAMPLES_LEN * TEST_SPLIT / 10)
val_split_at = int(EXAMPLES_LEN * TEST_SPLIT / 10 +
                   EXAMPLES_LEN * VAL_SPLIT / 10)

# print(EXAMPLES_LEN, val_split_at, test_split_at)

values_test = df_strings.iloc[indices[:test_split_at]]
values_val = df_strings.iloc[indices[test_split_at:val_split_at]]
values_train = df_strings.iloc[indices[val_split_at:]]

test = df2.iloc[getMorphemeBasedIndeicies(indices[:test_split_at])]
val = df2.iloc[getMorphemeBasedIndeicies(
    indices[test_split_at:val_split_at])]
train = df2.iloc[getMorphemeBasedIndeicies(indices[val_split_at:])]

# train["embeddings"]

# rand=np.random.randint(0,len(values_val))
# print(rand, values_test.iloc[rand])
# test["QApos"].iloc[rand*5:rand*5+5].idxmax(axis=1)

print(EXAMPLES_LEN, TEST_SPLIT / 10, VAL_SPLIT / 10,
      test_split_at, val_split_at)
print(values_test.shape, values_val.shape, values_train.shape)
print(test.shape, val.shape, train.shape)
print(test.shape[0] // 5)

# TEST = np.random.randint(0, len(values_train.index))
# print(TEST)
# print(values_train.loc[values_train.index[TEST], ["QAutf8"]])
# print(train.loc[train.index[TEST], "QAutf8"])

# TEST = np.random.randint(0, len(train.index))
# TEST = train.index[TEST]
# print(values_train.loc[TEST,["QAutf8"]])

# train.loc[TEST,:]

# arr = train.loc[TEST,:]
# arr.to_frame().T
# arr.index[arr == 1]
# pretty_join(train.loc[train.index[TEST], :])

# len(values_train) * 5, len(train)

embeddingInputLists = {
    i: [i + "_" + str(x) for x in embeddingInputSets[i]]
    for i in feat_x
}

data = {
    'input':
    getData(train, values_train, cats_feats=feat_x, strs_feats=strings_x,
            embeddings=WORD_EMB),
    'output':
    getData(train, values_train, cats_feats=feat_y, strs_feats=strings_y),
    'val': (
        getData(val, values_val, cats_feats=feat_x, strs_feats=strings_x,
             embeddings=WORD_EMB),
        getData(val, values_val, cats_feats=feat_y, strs_feats=strings_y)
    ),
    'test': (
        getData(test, values_test, cats_feats=feat_x, strs_feats=strings_x,
                embeddings=WORD_EMB),
        getData(
            test,
            values_test,
            cats_feats=feat_y,
            strs_feats=strings_y)
    )
}

# 8. Some info about shapes
print('\nTraining Data:')
print("\n".join(["X: " + i + str(data["input"][i].shape) for i in feat_x]))
print("\n".join(
    ["Y: " + i + str(data["output"][i].shape) for i in feat_y]))
print('\nValidation Data:')
print("\n".join(
    ["X: " + i + str(data["val"][0][i].shape) for i in feat_x]))
print("\n".join(
    ["Y: " + i + str(data["val"][1][i].shape) for i in feat_y]))
print('\nTest Data:')
print("\n".join(
    ["X: " + i + str(data["test"][0][i].shape) for i in feat_x]))
print("\n".join(
    ["Y: " + i + str(data["test"][1][i].shape) for i in feat_y]))

# # 7. Load Previous model

# model = load_model(MODEL_NAME)

# # 8. Build Model

print('Build model...')
outputs = []
inputs = []
# For strings
strings_input = layers.Input(
    shape=(STRING_LENGTH, len(charset)), name=strings_x[0])
lstm_strings_encoder = layers.Bidirectional(
    RNN(HIDDEN_SIZE, name="lstm_strings_encoder"))(strings_input)

# For categoricals
for i in feat_x:
    inputs.append(
        layers.Input(shape=(SENTLEN, data["input"][i].shape[2]), name=i))

# main_input = layers.Concatenate()(
#                     [layers.Dropout(0.1)(input) for input in inputs])

concatenatedInputs = [lstm_strings_encoder]
if len(inputs) >= 1:
    main_input = layers.Concatenate()(
        [getEmbedding(input) for input in inputs])

    lstm_out = layers.Bidirectional(RNN(HIDDEN_SIZE))(main_input)
    concatenatedInputs.append(lstm_out)
# input_shape=(None, len(ctable_x.chars) + EMBEDDINGS)))
# As the decoder RNN's input, repeatedly provide with last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
inputs.append(strings_input)
if WORD_EMB:
    emb_input = layers.Input(shape=(emb.vector_size,),
                             name=strings_x[0] + "_emb")
    inputs.append(emb_input)
    concatenatedInputs.append(emb_input)

if len(concatenatedInputs) > 1:
    concatenated = layers.Concatenate()(concatenatedInputs)
else:
    concatenated = concatenatedInputs[0]

# For strings again
repeat_strings_out = layers.RepeatVector(STRING_LENGTH)(concatenated)
rnn_out = RNN(HIDDEN_SIZE, return_sequences=True)(repeat_strings_out)
strings_output = layers.TimeDistributed(
    layers.Dense(len(charset), activation="softmax"),
    name=strings_y[0])(rnn_out)
outputs.append(strings_output)

dropout_out = layers.Dropout(0.5)(concatenated)
repeat_out = layers.RepeatVector(SENTLEN)(dropout_out)
# The decoder RNN could be multiple layers stacked or a single layer.
rnn_out = RNN(HIDDEN_SIZE, return_sequences=True)(repeat_out)
# By setting return_sequences to True, return not only the last output but
# all the outputs so far in the form of (num_samples, timesteps,
# output_dim). This is necessary as TimeDistributed in the below expects
# the first dimension to be the timesteps.
for _ in range(LAYERS - 1):
    rnn_out = RNN(HIDDEN_SIZE, return_sequences=True)(rnn_out)

# Apply a dense layer to the every temporal slice of an input. For each of
# step of the output sequence, decide which character should be chosen.

for i in feat_y:
    outputs.append(
        layers.TimeDistributed(
            layers.Dense(data["output"][i].shape[2], activation="softmax"),
            name=i)(rnn_out))

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# print(inputs,outputs)

# # 9. Summary

print(model.summary())

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
plot_model(
    model, to_file='plots/model.{}.png'.format(thedate), show_shapes=True)
# SVG(model_to_dot(model, show_shapes=True, show_layer_names=True
#    ).create(prog='dot', format='svg'))

# # 10. Training

earlyStopping = EarlyStopping(
    monitor='val_loss', patience=5, verbose=0, mode='auto')

tensorboard = TensorBoard(log_dir="logs/" + NAME + "_{}".format(thedate))

[x for x in model.metrics_names if x[-4:] == '_acc']

history = model.fit(
    data['input'],
    data['output'],
    batch_size=BATCH_SIZE,
    callbacks=[earlyStopping,
               TestCallback(data['test']), tensorboard],
    epochs=EPOCHS,
    verbose=2,
    validation_data=data['val'])

# model.metrics_names
# model.output_names
# model.metrics
# history.history

# summarize history for loss
fig = plt.figure()
with open('history/' + thedate + '.trainHistoryDict.pickle',
          'wb') as file_pi:
    pickle.dump(history.history, file_pi)

plt.plot(history.history["loss"])
plt.plot(history.history['val_loss'])
plt.plot(history.history['test_loss'])
plt.title('model overall loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'test'], loc='upper right')
filename = 'plots/model_overall_loss' + thedate + '.png'
print("PLOTTING: " + filename)
plt.draw()
plt.savefig(filename)
plt.show()
plt.close(fig)

# summarize history for loss
fig = plt.figure()
plt.plot(history.history["train_acc"])
plt.plot(history.history['val_acc'])
plt.plot(history.history['test_acc'])
plt.title('model average accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(
    [
        x + " = " + str(round(history.history[x + '_acc'][-1] * 100, 2))
        for x in ['train', 'val', 'test']
    ],
    loc='lower right')
# for i, x in enumerate(['train', 'val', 'test']):
#     plt.annotate(str(round(history.history[x+'_acc'][-1]*100,2)),
#                  xy=(len(history.history[x+'_acc']),
#                      history.history[x+'_acc'][-1]),
#                  textcoords='figure pixels',
#                  xytext=(-20,-10))
filename = 'plots/model_average_accuracy' + thedate + '.png'
print("PLOTTING: " + filename)
plt.draw()
plt.savefig(filename)
plt.show()
plt.close(fig)

# VALIDATAION = True
# prefix = "val_" if VALIDATAION else ""
# del prefix
fig = plt.figure()
legends = []

for x in model.output_names:
    # summarize history for accuracy
    plt.plot(history.history[x + "_acc"])
    legends.append("" + x + " = " +
                   str(round(history.history[x + '_acc'][-1] * 100, 2)))
#     plt.plot(history.history[x+"_acc"])
#     legends.append("val_"+x)
#     plt.plot(history.history["val_" + x+"_acc"])
#     legends.append("train_"+x)
plt.title('model indiviual accuracy on test dataset')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(legends, loc='lower right')
filename = 'plots/accuracy_indiv_all' + thedate + '.png'
print("PLOTTING: " + filename)
plt.draw()
plt.savefig(filename)
plt.show()
plt.close(fig)

fig = plt.figure()
legends = []
for x in model.output_names:
    # summarize history for loss
    plt.plot(history.history["test_" + x + "_loss"])
    legends.append("" + x + " = " +
                   str(round(history.history[x + '_loss'][-1] * 100, 2)))
plt.title('model individual loss on test dataset')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(legends, loc='upper right')
filename = 'plots/loss_all' + thedate + '.png'
print("PLOTTING: " + filename)
plt.draw()
plt.savefig(filename)
plt.show()
plt.close(fig)

# ## 10.1 Inspect One

# ## 10.2 Inspect All

mydata = data["test"]
preds = model.predict(mydata[0])
preds = [np.argmax(x, axis=-1) for x in preds]

def pretty_join2(arr, columns, v):
    return "/".join(
        [columns[row].replace(v + "_", "") for row in arr]).rstrip("/0")

# for ite in range(len(mydata[0][strings_x[0]])):
for ite in range(100):
    print("\nPredicted String Q",
          ctable.decode(mydata[0][strings_x[0]][ite], calc_argmax=True),
          "from", "-".join(str(x) for x in values_test.index.values[ite]))

    for ii, v in enumerate(model.output_names):
        if v not in strings_y:
            continue

        if not (np.argmax(mydata[1][v][ite],
                          axis=-1) == preds[ii][ite]).all():
            print('❌' + v)
            isAllCorrect = False
            print(
                "    T",
                utf2bw(ctable.decode(mydata[1][v][ite], calc_argmax=True)))
            print("     ",
                  utf2bw(ctable.decode(preds[ii][ite], calc_argmax=False)))

    rowy = dict()
    for ii, v in enumerate(model.output_names):
        if v in strings_y:
            continue
        rowy[v] = {"correct": np.argmax(mydata[1][v][ite], axis=-1)}
        rowy[v]["pred"] = preds[ii][ite]
        results = []
        #         print(val[v].columns)
        if not (rowy[v]["correct"] == rowy[v]["pred"]).all():
            isAllCorrect = False
            print('❌' + v, end=' ')
            #             results.append('☒')
            results.append(
                'T ' + pretty_join2(rowy[v]["correct"], val[v].columns, v))
            results.append(
                pretty_join2(rowy[v]["pred"], val[v].columns, v))
            print(' '.join(results))
#         elif v =="QApos":
        else:
            results.append(
                '✅' + 'T ' +
                pretty_join2(rowy[v]["correct"], val[v].columns, v))
            results.append(
                pretty_join2(rowy[v]["pred"], val[v].columns, v))
            print(v, ' '.join(results))

# rowy

# # 11. Save Model?

model.save("models/" + NAME + "{}.keras".format(thedate))
