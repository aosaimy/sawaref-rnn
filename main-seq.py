# -*- coding: utf-8 -*-
'''
    An implementation of sequence to sequence learning
    for performing ensemble morphosyntactic analyses
'''
from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
from prepare_data import SawarefData
from vis import SawarefVis
from character_table import colors, CharacterTable


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

sawarefData = SawarefData(MYPATH, EMBEDDINGS)

questions, expected, _, SENTLEN = sawarefData.get2DMorphemeJoinedFeatures(
    REVERSE, skipNAs=False)
questions = sawarefData.removeAlignment(questions, SENTLEN)
# questions_padded = pad_sequences(questions)
# expected_padded = pad_sequences(expected)

ctable_x = CharacterTable(
    set("-").union(set([xx for x in questions for xx in x])))

ctable_y = CharacterTable(
    set("-").union(set([xx for x in expected for xx in x])))

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
# MAXLEN = DIGITS + 1 + DIGITS


print('Total ayat questions:', len(questions))

print('Vectorization...')
x = np.zeros((len(questions), SENTLEN,
              len(ctable_x.chars)), dtype=np.bool)
# len(ctable_x.chars) + EMBEDDINGS), dtype=np.bool)
y = np.zeros((len(expected), SENTLEN,
              len(ctable_y.chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable_x.encode(sentence, SENTLEN)
    # x[i] = np.concatenate((ctable_x.encode([sentence], SENTLEN),
    # np.array([embeddings[i]])), 1)
for i, sentence in enumerate(expected):
    y[i] = ctable_y.encode(sentence, SENTLEN)

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
    input_shape=(None, len(ctable_x.chars))))
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
    layers.Dense(len(ctable_y.chars))))
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

SawarefVis(y_actual, y_pred,
           ctable_y.chars)
