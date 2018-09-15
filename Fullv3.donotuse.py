
# coding: utf-8

# ## Change Log (v.3)
# - Uses sample_weight to mask padded morphemes
# - sample_weight works for validation but not in loss function!
# - word_based accuracy. Done!
# - Supports experiment-based options, data and models
# 

# # 1. Import

# In[1]:


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
import datetime
from keras.callbacks import TensorBoard
from buckwalter import utf2bw, bw2utf
from pprint import pprint

from ws_client import WebSocketClient
# do not import in interactive mode
# from vis import SawarefVis
from keras.models import Sequential, Model, load_model
from keras import layers
from keras.callbacks import EarlyStopping, Callback
from keras.utils import plot_model

np.random.seed(0)

# get_ipython().magic('load_ext autoreload')
import sys
if len(sys.argv) < 2:
    print("Error: Please provide the experiment type")
    exit()

# # 2. Constants

# In[300]:

global_emb = None
class Experiment:
    global global_emb 
    MYPATH = "/morpho/output/"
    # Parameters for the model and dataset.
    TRAINING_SIZE = 50000
    EPOCHS = 100
    EMBEDDINGS = 0
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
    CATS_EMBEDDING = 1000
    TEST_SPLIT = 1  #  means 0.1 of data is for test
    VAL_SPLIT = 1  #  means 0.1 of data is for test
    LOAD_FASTTEXT = False
    CHARACTER_BASED = False
    MORPHEME_BASED = False
    previous_names = set()
    emb = global_emb
    
    def __init__(self, exp_type="e"):
        self.exp_type = exp_type
        self.set_constants()
        self.set_name()
        self.load_emb()
    
    def load_emb(self):
        # global global_emb
        if self.LOAD_FASTTEXT:
            # from gensim.models import FastText
            # if global_emb == None:
            self.emb = WebSocketClient("ws://localhost:8765/")

    def set_name(self):
        name = input("What is the name of this experiment? (type="+self.exp_type+")") if len(sys.argv) < 2 else sys.argv[1]
        self.NAME = self.exp_type+"_"+name
        self.thedate = datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M")
        print("\nNAME=", self.NAME)
        print("DATE=", self.thedate)
        self.previous_names.add(self.NAME+"%"+self.thedate)
    
    def set_constants(self):
        self.MYPATH = "/morpho/output/"
        # Parameters for the model and dataset.
        self.TRAINING_SIZE = 50000
        self.EPOCHS = 40
        self.EMBEDDINGS = 0
        # DIGITS = 3
        # REVERSE = True
        # Try replacing GRU, or SimpleRNN.
        self.HIDDEN_SIZE = 128
        self.BATCH_SIZE = 64
        self.LAYERS = 1
        self.ITERATIONS = 10
        self.MODEL_NAME = "main-seq-multiinput-multioutput-segmentation.keras"
        self.DATA_PICKLE = "main-seq-multiinput-multioutput-segmentation.pickle"
        self.RNN = layers.LSTM
        if self.exp_type == "comp_ru" or self.exp_type == "comp_sp1":
            self.CATS_EMBEDDING = 1000
            self.TEST_SPLIT = 1  #  means 0.1 of data is for test
            self.VAL_SPLIT = 1  #  means 0.1 of data is for test
            self.LOAD_FASTTEXT = True
            self.CHARACTER_BASED = False
            self.MORPHEME_BASED = True
            self.ALIGN_TYPE = "_ru" if self.exp_type == "comp_ru" else "_sp1"
        elif self.exp_type == "comp_ch":
            self.CATS_EMBEDDING = 1000
            self.TEST_SPLIT = 1  #  means 0.1 of data is for test
            self.VAL_SPLIT = 1  #  means 0.1 of data is for test
            self.LOAD_FASTTEXT = False
            self.CHARACTER_BASED = False
            self.MORPHEME_BASED = False
            self.ALIGN_TYPE = "_ch"
        elif self.exp_type == "comp_end":
            self.CATS_EMBEDDING = 1000
            self.TEST_SPLIT = 1  #  means 0.1 of data is for test
            self.VAL_SPLIT = 1  #  means 0.1 of data is for test
            self.LOAD_FASTTEXT = True
            self.CHARACTER_BASED = False
            self.MORPHEME_BASED = False
            self.ALIGN_TYPE = ""
        else:
            print("ERROR: Not a valid config name")
            raise NameError("ERROR: Not a valid config name")


e = Experiment("comp_end")


# ## Experiemnts Names
# ### Compartative: 
# - Morpheme-based (ru, sp1,),  `comp_ru`, `comp_sp1`, 
# - character-based, `comp_ch`
# - end-to-end `comp_end`, 
# 

# # 3. Config

# `feat_x` is the input categorical features
# 
# `feat_y` is the output categorical features
# 
# `strings_x` is the input character-based strings features
# 
# `strings_y` is the output character-based strings features

# In[4]:

feat_x = [
    "MXpos", "STpos", "AMpos", "FApos", "STaspect", "AMaspect", "MXaspect",
    "FAaspect", "STperson", "AMperson", "MXperson", "FAperson", "STgender",
    "AMgender", "MXgender", "FAgender", "STnumber", "AMnumber", "MXnumber",
    "FAnumber", "STcase", "AMcase", "MXcase", "FAcase", "STvoice", "AMvoice",
    "MXvoice", "FAvoice", "STmood", "AMmood", "MXmood", "FAmood", "STstate",
    "AMstate", "MXstate", "FAstate"
]
feat_y = [
    "QApos", "QAaspect", "QAperson", "QAgender", "QAnumber", "QAcase",
    "QAvoice", "QAmood", "QAstate"
]
strings_x = ["QAwutf8", "word"]
strings_y = ["QAutf8"]
if e.CHARACTER_BASED:
    strings_x = ["bw"]
elif e.MORPHEME_BASED:
    strings_x = []


accuracies = pd.DataFrame([],
             columns=["type"]+feat_y+strings_y+["agg"])

# In[286]:

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


#     return df.values.reshape((df.shape[0]//middle_dim, middle_dim, df.shape[1]))


def flattencolumns(df1, cols):
    df = pd.concat(
        [pd.DataFrame(df1[x].values.tolist()).add_prefix(x) for x in cols],
        axis=1)
    return pd.concat([df, df1.drop(cols, axis=1)], axis=1)


def truncate(x):
    return x[:EMBEDDINGS]


def removeDiac(x):
    return re.sub('[ًٌٍَُِّْ��]', '', x.replace("ٱ","ا"))


def padStringWithSpaces(x):
    return x + ' ' * (e.STRING_LENGTH - len(x))


def joinMorphemesStrings(arr):
    return "+".join([
        x for x in arr if isinstance(x, float) == False and x != -1
        and x != "-----" and x != "-"
    ])


def fullprint(*args, **kwargs):
    opt = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)
    

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        metrics = self.model.evaluate(x, y, verbose=0)
        for i,x in enumerate(self.model.metrics_names):
            logs["test_"+x]= metrics[i]
        logs['test_acc']= np.mean([logs["test_"+x] for x in self.model.metrics_names if x[-4:]=='_acc'])
        logs['val_acc']= np.mean([logs["val_"+x] for x in self.model.metrics_names if x[-4:]=='_acc'])
        logs['train_acc']= np.mean([logs[x] for x in self.model.metrics_names if x[-4:]=='_acc'])
        return logs

def calc_accuracy(name, model,mydata,data_length, debug=True):
    results = pd.DataFrame([], columns=[[x for x in model.output_names*3]+["agg"],[x for x in ["acc","pred","actu"]*len(model.output_names)]+["agg"]])
    results.sort_index(axis=1, inplace=True)
    for ite in range(data_length):
        r = dict()
        for i, v in enumerate(model.output_names):
            if v not in strings_y:
                continue

            r[(v,"acc")] = (np.argmax(mydata[1][v][ite], axis=-1) == preds[i][ite]).all()
            r[(v,"pred")] = utf2bw(ctable.decode(preds[i][ite], calc_argmax=False)).strip(" ") if debug else ""
            r[(v,"actu")] = utf2bw(ctable.decode(mydata[1][v][ite], calc_argmax=True)).strip(" ") if debug else ""

        for i, v in enumerate(model.output_names):
            if v in strings_y:
                continue
            correct, pred = np.argmax(mydata[1][v][ite], axis=-1), preds[i][ite]
            r[(v,"acc")] = (correct == pred).all()
            r[(v,"actu")] = pretty_join2(correct,val[v].columns,v) if debug else ""
            r[(v,"pred")] = pretty_join2(pred,val[v].columns,v) if debug else ""
        r[("agg","agg")] = sum([r[(v,"acc")] for i, v in enumerate(model.output_names)]) == len(model.output_names)
        results.loc[ite] = r
    accuracies.loc[len(accuracies)] = [name]+list(np.average(results.filter(regex="acc|agg").values.astype(int),axis=0))
    return results

# # 4. Loading data from sawaref

# In[7]:

sawarefData = SawarefData(
    e.MYPATH,
    e.EMBEDDINGS,
    align_type=e.ALIGN_TYPE,
    feat_x=feat_x,
    strings_x=strings_x,
    strings_y=strings_y,
    feat_y=feat_y)


# In[8]:

source = list(itertools.chain(*sawarefData.quran_sent))
df = pd.DataFrame(
    source,
    columns=["sid", "aid", "wid", "mid"] + feat_x + strings_x + strings_y +
    ["embeddings"] + feat_y)
if e.EMBEDDINGS > 0:
    df["embeddings"] = df["embeddings"].apply(truncate)
    df = flattencolumns(df, ["embeddings"])
df.set_index(["sid", "aid", "wid", "mid"], inplace=True)
df.sort_index(inplace=True)


# In[9]:

strings = strings_x + strings_y

# a. clean all padded rows
for s in strings:
    #df.loc[df2[("vals", s)] == -1, ("vals", s)] = -1
    df.loc[pd.isna(df[s]), s] = ""

for x in strings:
    df[x+"_undiac"] = df[x].apply(removeDiac)


# In[10]:

## 2. Pad the rows according to the longest word (in # of morphemes)
e.SENTLEN = max(df.index.get_level_values("mid"))
df = df.reindex(
    padIndexes(df, max(df.index.get_level_values("mid"))),
    fill_value=-1).sort_index()


# In[11]:

## 3. Get the hot encoding of all caterogirical data (see columns attr)
dumm = pd.get_dummies(df, columns=feat_x + feat_y)


# In[12]:

for x in feat_x:
    dumm.drop(x+"_-1",axis=1,inplace=True)


# In[14]:

dumm.drop("embeddings", axis=1, inplace=True)


# In[19]:

## 4. Add two-level columns for easy indexing later (wid, mid)
e.EXAMPLES_LEN = df.shape[0] // e.SENTLEN
new_columns = []
for x in dumm.columns:
    new_columns.append(re.sub('(_.*|[0-9]*)', '', x))
dumm.columns = [new_columns, dumm.columns]
dumm.index = [[x for x in range(e.EXAMPLES_LEN) for _ in range(e.SENTLEN)],
              [x for _ in range(e.EXAMPLES_LEN) for x in range(e.SENTLEN)]]
dumm.sort_index(axis=1, inplace=True)


# In[20]:

## 5. Find possible values of each cat
def getSet(df):
    results = set()
    df.apply(results.add)
    return results


embeddingInputSets = {i: getSet(df[i]) for i in feat_x}


# In[21]:

feat_x = list(set(feat_x) - set([x for x in feat_x if len( embeddingInputSets[x])<=1]))


# In[22]:

df2 = pd.concat([df.reset_index(), dumm.reset_index()], axis=1)
# df2.set_index(["sid", "aid", "wid", "mid"], inplace=True)
df2.index = [[x for x in range(e.EXAMPLES_LEN) for _ in range(e.SENTLEN)],
             [x for _ in range(e.EXAMPLES_LEN) for x in range(e.SENTLEN)]]

df2.drop(['embeddings' + str(x) for x in range(e.EMBEDDINGS)], inplace=True, axis=1)
# df2.columns = [(x,"val") if isinstance(x,str) else x  for x in df2.columns]


# In[23]:

df2.index = [[x for x in range(e.EXAMPLES_LEN) for _ in range(e.SENTLEN)],
             [x for _ in range(e.EXAMPLES_LEN) for x in range(e.SENTLEN)]]


# In[24]:

df2.columns = [["vals" if isinstance(x, str) else x[0] for x in df2.columns],
               [x if isinstance(x, str) else x[1] for x in df2.columns]]

df2.sort_index(axis=1, inplace=True)


# In[25]:

del df
del dumm


# In[53]:

def emb_encode(x):
    return np.zeros(e.emb.vector_size) if re.sub("\\s*","",x)=="" or x == -1 else e.emb[x]

def convertStringToFeatX(x):
    df2[(x,"emb")] = df2[(x,x+"_undiac")].apply(emb_encode)
    return df2.join(pd.DataFrame(np.stack(df2[(x,"emb")].values),
                         index=df2.index, 
                         columns=[[x+"_emb" for i in range(emb.vector_size)],
                                  [x+"_emb_"+str(i) for i in range(emb.vector_size)]]))

if not e.CHARACTER_BASED and e.MORPHEME_BASED:
    df2 = convertStringToFeatX("word")
    feat_x.append("word_emb")


# In[27]:

# b. group them by morpheme and join with "+""
df_strings = pd.DataFrame({
    x: df2["vals", x].groupby(level=[0]).apply(joinMorphemesStrings)
    for x in strings
})


# In[31]:

# c. pad joined morphemes
e.STRING_LENGTH = max([len(x) for k in strings for x in df_strings[k]])
for s in strings:
    df_strings[s] = df_strings[s].apply(padStringWithSpaces)


# In[32]:

# d. encode them in one hot encoding
charset = set("+").union(
    *[list(set("".join(df_strings[x] + "-"))) for x in strings])
ctable = CharacterTable(charset, e.STRING_LENGTH)
### Now we have one shape for all strings: (STRING_LENGTH, len(charset))
for x in strings:
    df_strings[x + "_onehot"] = df_strings[x].apply(ctable.encode)
df_strings['num'] = [x for x in range(len(df_strings))]
df_strings.set_index('num', append=True, inplace=True)


# In[33]:

# e. remove diac
for x in strings:
    df_strings[x+ "_undiac"] = df_strings[x].apply(removeDiac)


# In[54]:

# f. encode them as dense vector using fastText
for x in strings:
    if e.LOAD_FASTTEXT:
        df_strings[x + "_emb"] = df_strings[x+"_undiac"].apply(emb_encode)


# # 6. Prepare splits

# In[62]:

# 6. Shuffle (x, y) in unison
indices = list(range(e.EXAMPLES_LEN))
np.random.shuffle(indices)


# In[63]:

# 7. Explicitly set apart 10% for validation data that we never train over.
test_split_at = int(e.EXAMPLES_LEN * e.TEST_SPLIT / 10)
val_split_at = int(e.EXAMPLES_LEN * e.TEST_SPLIT / 10 +
                   e.EXAMPLES_LEN * e.VAL_SPLIT / 10)


# In[64]:

def getMorphemeBasedIndeicies(arr):
    return np.array([list(range(x*e.SENTLEN,x*e.SENTLEN+e.SENTLEN)) for x in arr]).flatten()


# In[65]:

values_test = df_strings.iloc[indices[:test_split_at]]
values_val = df_strings.iloc[indices[test_split_at:val_split_at]]
values_train = df_strings.iloc[indices[val_split_at:]]

test = df2.iloc[getMorphemeBasedIndeicies(indices[:test_split_at])]
val = df2.iloc[getMorphemeBasedIndeicies(indices[test_split_at:val_split_at])]
train = df2.iloc[getMorphemeBasedIndeicies(indices[val_split_at:])]


# In[66]:

embeddingInputLists = {
    i: [i + "_" + str(x) for x in embeddingInputSets[i]]
    for i in embeddingInputSets
}


# In[189]:

def getValuesAndReshape(df, middle_dim, test=""):
    return df.values.reshape((df.shape[0] // middle_dim, middle_dim, -1))

def setZero(df,zeros, from_val=-1, to_val=0.):
#     dumm[(dumm.isin([0,-1])).all(axis=1)] = -1.
    if zeros:
        df[(df==from_val).all(axis=1)] = to_val
    return df

def getData(cat_source, str_source, cats_feats=[], strs_feats=[], embeddings=False, zeros= False):
    data = {
        **{i:
           getValuesAndReshape(cat_source[i].idxmax(axis=1).apply(lambda x: embeddingInputLists[i].index(x)),e.SENTLEN)
               if i in embeddingInputSets and len(embeddingInputSets[i]) > 10000+e.CATS_EMBEDDING #and i.replace("_emb","") not in strings_x
               else getValuesAndReshape(cat_source[i], e.SENTLEN,i) 
           for i in cats_feats},
        **{i:np.stack(str_source[i+"_onehot"].values) for i in strs_feats},
    }
    if embeddings:
        for i in strs_feats:
            if i+"_emb" not in data:
                data[i+"_emb"] = np.stack(str_source[i+"_emb"].values)
    if zeros:
        for i in cats_feats:
            data[i][np.where(data[i]==-1.)] = 0.
    return data


data = {
    'input': getData(train, values_train, cats_feats=feat_x, strs_feats=strings_x, embeddings=e.LOAD_FASTTEXT),
    'output': getData(train, values_train, cats_feats=feat_y, strs_feats=strings_y, zeros=True),
    'val': (
        getData(val, values_val, cats_feats=feat_x, strs_feats=strings_x, embeddings=e.LOAD_FASTTEXT),
        getData(val, values_val, cats_feats=feat_y, strs_feats=strings_y, zeros=True)
    ),
    'test': (
        getData(test, values_test, cats_feats=feat_x, strs_feats=strings_x, embeddings=e.LOAD_FASTTEXT),
        getData(test, values_test, cats_feats=feat_y, strs_feats=strings_y, zeros=True)
    )

}
data["val"] = tuple([*data["val"],
                     {i: (data["val"][1][i].argmax(axis=2) > 0).astype(int) for i in feat_y + strings_y if i in data["val"][1]},]
                   )


# # 8. Build Model

# ## 8.1 End-To-End

# In[73]:

if not e.MORPHEME_BASED and not e.CHARACTER_BASED:
    # print('Build model...')
    outputs = []
    inputs = []
    # For strings
    strings_input = layers.Input(shape=(e.STRING_LENGTH, len(charset)), name=strings_x[0])
    lstm_strings_encoder = layers.Bidirectional(e.RNN(e.HIDDEN_SIZE,name="lstm_strings_encoder"))(strings_input)

    # For categoricals
    for i in feat_x:
        inputs.append(layers.Input(shape=(e.SENTLEN, data["input"][i].shape[2]), name=i))

    def getEmbedding(input):
        name = input.name.split("_")[0]
        if name in feat_x and len(embeddingInputSets[name]) > e.CATS_EMBEDDING:
            return layers.Masking(mask_value=0.)(layers.Reshape((e.SENTLEN, -1))(layers.Embedding(len(embeddingInputSets[name]),2, input_length=e.SENTLEN)(input)))
        else:
            return layers.Dropout(0.1)(layers.Masking(mask_value=0.)(input))

    # main_input = layers.Concatenate()([layers.Dropout(0.1)(input) for input in inputs])
    main_input = layers.Concatenate()([getEmbedding(input) for input in inputs])
    inputs.append(strings_input)

    lstm_out = layers.Bidirectional(e.RNN(e.HIDDEN_SIZE))(main_input)
    # input_shape=(None, len(ctable_x.chars) + EMBEDDINGS)))
    # As the decoder e.RNN's input, repeatedly provide with the last hidden state of
    # e.RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    if e.LOAD_FASTTEXT:
        emb_input = layers.Input(shape=(e.emb.vector_size,), name=strings_x[0]+"_emb")
        inputs.append(emb_input)
        concatenated = layers.Concatenate()([lstm_strings_encoder,lstm_out,emb_input])
    else:
        concatenated = layers.Concatenate()([lstm_strings_encoder,lstm_out])

    # For strings again
    repeat_strings_out = layers.RepeatVector(e.STRING_LENGTH)(concatenated)
    rnn_out = e.RNN(e.HIDDEN_SIZE, return_sequences=True)(repeat_strings_out)
    strings_output = layers.TimeDistributed(
          layers.Dense(
            len(charset), 
            activation="softmax"), name=strings_y[0])(rnn_out)
    outputs.append(strings_output)

    dropout_out = layers.Dropout(0.5)(concatenated)
    repeat_out = layers.RepeatVector(e.SENTLEN)(dropout_out)
    # The decoder e.RNN could be multiple layers stacked or a single layer.
    rnn_out = e.RNN(e.HIDDEN_SIZE, return_sequences=True)(repeat_out)
    for _ in range(e.LAYERS-1):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        rnn_out = e.RNN(e.HIDDEN_SIZE, return_sequences=True)(rnn_out)



    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.

    for i in feat_y:
        outputs.append(
          layers.TimeDistributed(
          layers.Dense(
            data["output"][i].shape[2], 
            activation="softmax"), name=i)(rnn_out))


# ## 8.2 Morpheme-Based  or 8.3 Character-Based 

# In[74]:

if e.MORPHEME_BASED or e.CHARACTER_BASED:
    # print('Build model...')
    outputs = []
    inputs = []

    # For categoricals
    for i in feat_x:
        inputs.append(layers.Input(shape=(SENTLEN, data["input"][i].shape[2]), name=i))

    def getEmbedding(input):
        name = input.name.split("_")[0]
        if name in feat_x and len(embeddingInputSets[name]) > CATS_EMBEDDING:
            return layers.Masking(mask_value=0.)(layers.Reshape((SENTLEN, -1))(layers.Embedding(len(embeddingInputSets[name]),2, input_length=SENTLEN)(input)))
        else:
            return layers.Dropout(0.1)(layers.Masking(mask_value=0.)(input))

    main_input = layers.Concatenate()([getEmbedding(input) for input in inputs])

    lstm_out = layers.Bidirectional(RNN(HIDDEN_SIZE, return_sequences=True))(main_input)


    rnn_out = lstm_out



    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.

    for i in feat_y:
        outputs.append(
          layers.TimeDistributed(
          layers.Dense(
            data["output"][i].shape[2], 
            activation="softmax"), name=i)(rnn_out))


# ## Compile

# In[200]:

model = Model(inputs=inputs, outputs=outputs)


model.compile(sample_weight_mode="temporal" if not e.MORPHEME_BASED and not e.CHARACTER_BASED else None,
              loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
            )


# # 9. Summary

# In[508]:

model.summary()


# In[77]:

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
plot_model(model, to_file='model_only_emb.png', show_shapes=True)
SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))


# # 10. Training

# In[203]:

earlyStopping=EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

tensorboard = TensorBoard(
    log_dir="logs/" + e.NAME + "_{}".format(e.thedate))


# In[285]:

print(e.NAME + "_{}".format(e.thedate))
history = model.fit(data['input'], data['output'],
                    batch_size=e.BATCH_SIZE,
                    callbacks=[earlyStopping, TestCallback(data['test']), tensorboard],
                    epochs=e.EPOCHS,
                    verbose=2,
#                     sample_weight={i:(data["output"][i].argmax(axis=2) > 0).astype(int) for i in model.output_names},
                    validation_data=data['val'])


# In[ ]:
run_name = input("name this run") if len(sys.argv) < 3 else sys.argv[2]

print(calc_accuracy(e.NAME+run_name, model, mydata, len(mydata[0][strings_x[0]]),debug=False))


# In[ ]:

print(accuracies)


# In[ ]:

fig = plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history['val_loss'])
plt.plot(history.history['test_loss'])
plt.title('model overall loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'test'], loc='upper right')
plt.savefig('plots/model_overall_loss' +
            datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M") + '.png')
plt.close(fig)

# plt.show()


# In[471]:

# summarize history for loss
plt.plot(history.history["train_acc"])
plt.plot(history.history['val_acc'])
plt.plot(history.history['test_acc'])
plt.title('model average accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend([x +" = "+ str(round(history.history[x+'_acc'][-1]*100,2)) for x in ['train', 'val', 'test']], loc='lower right')
# for i, x in enumerate(['train', 'val', 'test']):
#     plt.annotate(str(round(history.history[x+'_acc'][-1]*100,2)),
#                  xy=(len(history.history[x+'_acc']), history.history[x+'_acc'][-1]), 
#                  textcoords='figure pixels', 
#                  xytext=(-20,-10))
plt.savefig('plots/model_average_accuracy' +
            datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M") + '.png')


# In[ ]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history['val_loss'])
plt.plot(history.history['test_loss'])
plt.title('model overall loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'test'], loc='upper right')
plt.savefig('plots/model_overall_loss' +
            datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M") + '.png')

# plt.show()


# In[ ]:

# summarize history for loss
plt.plot(history.history["train_acc"])
plt.plot(history.history['val_acc'])
plt.plot(history.history['test_acc'])
plt.title('model average accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend([x +" = "+ str(round(history.history[x+'_acc'][-1]*100,2)) for x in ['train', 'val', 'test']], loc='lower right')
# for i, x in enumerate(['train', 'val', 'test']):
#     plt.annotate(str(round(history.history[x+'_acc'][-1]*100,2)),
#                  xy=(len(history.history[x+'_acc']), history.history[x+'_acc'][-1]), 
#                  textcoords='figure pixels', 
#                  xytext=(-20,-10))
plt.savefig('plots/model_average_accuracy' +
            datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M") + '.png')

plt.show()


# In[ ]:

# VALIDATAION = True
# prefix = "val_" if VALIDATAION else ""
# del prefix
legends = []
for x in model.output_names:
    # summarize history for accuracy
    plt.plot(history.history[x+"_acc"])
    legends.append(""+x +" = "+ str(round(history.history[x+'_acc'][-1]*100,2)))
#     plt.plot(history.history[x+"_acc"])
#     legends.append("val_"+x)
#     plt.plot(history.history["val_" + x+"_acc"])
#     legends.append("train_"+x)
plt.title('model indiviual accuracy on test dataset')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(legends, loc='lower right')
plt.savefig('plots/accuracy_all' +
            datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M") + '.png')
plt.show()


# In[ ]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

legends = []
for x in model.output_names:
    # summarize history for loss
    plt.plot(history.history["test_"+x+"_loss"])
    legends.append(""+x +" = "+ str(round(history.history[x+'_loss'][-1]*100,2)))
plt.title('model individual loss on test dataset')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(legends, loc='upper right')
plt.savefig('plots/loss_all' +
            datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M") + '.png')
plt.show()


# ## 10.1 Inspect One

# In[ ]:

#%%capture --no-stderr cap
get_ipython().magic('autoreload 2')

colors.ok = ''
colors.fail = ''
colors.close = ''


def inspectOne(times=0, printCorrect=True):
    isAllCorrect = True
    if times > 10000: # to prevent infinite loops
        return
    ind = np.random.randint(0, len(val.index))
    ind = (val.index[ind][0], slice(None))

    string_input = values_test.loc[(slice(None), ind[0]), :]
    preds = model.predict(
        getData(val.loc[ind], string_input, feat_x, strings_x))
    preds = [np.argmax(x, axis=-1) for x in preds]
    #predicted string
    print("Predicted String Q", string_input[strings_x[0]][0], "from",
          "-".join(str(x) for x in string_input.index.values[0]))

    if (np.argmax(string_input[strings_y[0]][0], axis=-1) == preds[0]).all():
        if printCorrect: print(colors.ok + '✅' + colors.close + "Segmentation")
    else:
        print(colors.fail + '❌' + colors.close + "Segmentation")
        isAllCorrect = False
        print("    T", utf2bw(string_input[strings_y[0]][0]))
        print("     ", utf2bw(ctable.decode(preds[0][0], calc_argmax=False)))


#     print('Q', utf2bw(pretty_join(rowx)))

    rowy = dict()
    for i, v in enumerate(feat_y):
        rowy[v] = {"correct": val[v].loc[ind]}
        res = np.zeros((SENTLEN, rowy[v]["correct"].shape[1]))
        for ii, c in enumerate(preds[i + 1][0]):
            res[ii, c] = 1
        rowy[v]["pred"] = pd.DataFrame(res, columns=val[v].columns)
        results = []
        if (rowy[v]["correct"].values == rowy[v]["pred"].values).all():
            if printCorrect: print(colors.ok + '✅' + colors.close + v)
        else:
            isAllCorrect = False
            print(colors.fail + '❌' + colors.close + v, end=' ')
            #             results.append(colors.fail + '☒' + colors.close)
            results.append('T ' + pretty_join(rowy[v]["correct"]))
            results.append(pretty_join(rowy[v]["pred"]))
            print(' '.join(results))
    if isAllCorrect or times < 10:
        print("")
        inspectOne(times + 1, printCorrect)

inspectOne(printCorrect=False)

with open(
        'output' + datetime.datetime.now().strftime(".%Y.%m.%d.%H.%M.%S") +
        '.txt', 'w') as f:
    f.write(cap.stdout)


# ## 10.2 Inspect All

# In[206]:

mydata = data["test"]
preds_org = model.predict(mydata[0])
preds = [np.argmax(x, axis=-1) for x in preds_org]


# In[194]:

train["QAaspect"].iloc[2]


# In[197]:

# len(preds)
model.output_names
preds[2]


# In[198]:

def pretty_join2(arr, columns, v):
    return "/".join([columns[row].replace(v+"_","") for row in arr]).rstrip("/0")


# In[207]:

# for ite in range(len(mydata[0][strings_x[0]])):
for ite in range(10):
    print("\nPredicted String Q", ctable.decode(mydata[0][strings_x[0]][ite], calc_argmax=True), "from",
          "-".join(str(x) for x in values_test.index.values[ite]))
    
    for i, v in enumerate(model.output_names):
        if v not in strings_y:
            continue

        if not (np.argmax(mydata[1][v][ite], axis=-1) == preds[i][ite]).all():
            print(colors.fail + '❌' + colors.close + v)
            isAllCorrect = False
            print("    T", utf2bw(ctable.decode(mydata[1][v][ite], calc_argmax=True)))
            print("     ", utf2bw(ctable.decode(preds[i][ite], calc_argmax=False)))

    rowy = dict()
    for i, v in enumerate(model.output_names):
        if v in strings_y:
            continue
        rowy[v] = {"correct": np.argmax(mydata[1][v][ite], axis=-1)}
        rowy[v]["pred"] = preds[i][ite]
        results = []
#         print(val[v].columns)
        if not (rowy[v]["correct"] == rowy[v]["pred"]).all():
            isAllCorrect = False
            print(colors.fail + '❌' + colors.close + v, end=' ')
            #             results.append(colors.fail + '☒' + colors.close)
            results.append('T ' + pretty_join2(rowy[v]["correct"],val[v].columns,v))
            results.append(pretty_join2(rowy[v]["pred"],val[v].columns,v))
            print(' '.join(results))
#         elif v =="QApos":
        else:
            results.append(colors.ok + '✅' + colors.close + 'T ' + pretty_join2(rowy[v]["correct"],val[v].columns,v))
            results.append(pretty_join2(rowy[v]["pred"],val[v].columns,v))
            print(v, ' '.join(results))


# In[ ]:

for _ in range(2):
    inspectOne(print)


# # 11. Save Model?

# In[ ]:

model.save("models/" + NAME + "{}.keras".format(thedate))


# In[ ]:

import importlib
importlib.reload(character_table)


# In[ ]:

ctable.decode(ctable.encode("ب"), calc_argmax=False)


# 
# ### After alignment. Accuracy is good. Can be treated as baseline. (name=baseline)
# `strings_cats_aligned_2018.06.25.15.14`
# ### Comaprison between baseline and POS embeddings. (No drop, so it is the best) (name=baseline+pos_emb)
# `with_pos_embeddings_2018.06.25.17.37`
# ### Comaprison between baseline with POS embeddings and subword embeddings. (name=baseline+pos_emb+subword_emb)
# `with_embeddings_2018.06.25.17.37` NOT DONE
# ### Comaprison between baseline with subword embeddings. (name=baseline+pos_emb+subword_emb+word2vec)
# `with_word2vec_2018.06.25.17.37` NOT DONE
# 
# ### Different Sizes of Baseline or subword
