# import random
from collections import Counter
import numpy as np
import pandas as pd
import json
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l1l2
from keras.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
from keras.optimizers import Adam

import sys
import os
import errno

sys.setrecursionlimit(10000)


# define fbeta beta fn
def fbeta_custom(x, y):
    return fbeta_score(x, y, beta=1.0)


# # to create dir if dir does not exist
def create_filepath(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def shuffle_X_Y(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


CLASSES = [
    'low',
    'mid',
    'high',
]

CLASS_BINS = [
    -10.0,
    4.0,
    8.0,
    100.0,
]

# define class weight to tackle skew
CLASS_WEIGHT = {
    'forward': {
        0: 1.,
        1: 5.3 * 0.6,
        2: 9.4 * 0.6,
    },
    'midfielder': {
        0: 1.,
        1: 13.8 * 0.6 * 0.7,
        2: 21.1 * 0.6 * 0.7 * 0.82,
    },
    'defender': {
        0: 1.,
        1: 5.58 * 0.6 * 0.7,
        2: 30.71 * 0.6 * 0.7 * 0.75,
    },
    'goalkeeper': {
        0: 1.,
        1: 10.,
        2: 11.,
    },
}


def get_class_counts(Y):
    y_1d = np.empty((Y.shape[0], 1), dtype=np.object_)
    i = 0
    for y_entry in Y:
        class_num = np.where(y_entry == 1)[0][0]
        y_1d[i] = CLASSES[class_num]
        i += 1

    y_1d = y_1d.ravel()
    unique, counts = np.unique(y_1d, return_counts=True)
    return dict(zip(unique, counts))


def apply_undersampling(X, Y, class_num=0, frac_removed=0.4):
    # # print stats before undersampling
    print('number of training samples before undersampling = %s' % X.shape[0])
    print('Y counter before undersampling -')
    print(get_class_counts(Y))

    # # get indices of rows to be removed for majority class
    y_0 = Y[:, class_num]
    low_indices = np.where(y_0 == 1)[0]
    n_removed = int(low_indices.shape[0] * frac_removed)

    # # delected removed rows
    removed = low_indices[0:n_removed]
    Y = np.delete(Y, removed, axis=0)
    X = np.delete(X, removed, axis=0)

    # # print stats after undersampling
    print('number of training samples after undersampling = %s' % X.shape[0])
    print('Y counter after undersampling -')
    print(get_class_counts(Y))

    return X, Y


def apply_smote(X, Y, ratio=1.0):
    print('number of training samples before smote = %s' % X.shape[0])
    sm = SMOTE(kind='regular', ratio=ratio)
    # ada = ADASYN(ratio=ratio)

    # convert y to 1d
    y_1d = np.empty((Y.shape[0], 1), dtype=np.object_)
    i = 0
    for y_entry in Y:
        class_num = np.where(y_entry == 1)[0][0]
        y_1d[i] = CLASSES[class_num]
        i += 1

    y_1d = y_1d.ravel()
    print(y_1d.shape)
    print(y_1d)
    print('Y counter before SMOTE -')
    unique, counts = np.unique(y_1d, return_counts=True)
    print(dict(zip(unique, counts)))
    X, Y = sm.fit_sample(X, y_1d)
    # X, Y = ada.fit_sample(X, y_1d)
    print('Y counter after 1st SMOTE -')
    print(Counter(Y))
    X, Y = sm.fit_sample(X, Y)
    print('Y counter after 2nd SMOTE -')
    print(Counter(Y))

    # one hot encode again
    i = 0
    y_temp = np.zeros(shape=(len(Y), len(CLASSES)))
    for y_entry in Y:
        class_num = CLASSES.index(y_entry)
        y_temp[i][class_num] = 1
        i = i + 1
    Y = y_temp
    print('number of training samples after smote = %s' % X.shape[0])
    return X, Y


# # method to create confusion matrix
def get_confusion_matrix_one_hot(model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results, axis=1)
    assert len(predictions) == truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:, actual_class] == 1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class == predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix) == len(truth)
    assert np.sum(confusion_matrix) == np.sum(truth)
    return confusion_matrix


# # define models
def three_layer_nn(num_of_features=0):
    # # create model
    K.set_learning_phase(1)
    model = Sequential()
    model.add(Dense(
        num_of_features,
        input_dim=num_of_features,
        W_regularizer=l1l2(0.1),
        init='normal',
        activation='relu'
    ))
    # # add dropout to regularize
    model.add(Dropout(0.35))
    # # add hidden layer
    model.add(Dense(num_of_features, init='normal', activation='relu'))
    # # add output layer
    model.add(Dense(len(CLASSES), init='normal', activation='softmax'))
    # # define loss optimiser
    adam = Adam(lr=0.0002)
    # # compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', fbeta_custom])
    return model


def four_layer_nn(num_of_features=0):
    # # create model
    K.set_learning_phase(1)
    model = Sequential()
    model.add(Dense(
        num_of_features,
        input_dim=num_of_features,
        W_regularizer=l1l2(0.1),
        init='normal',
        activation='relu'
    ))
    # model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Dense(int(num_of_features * 1.5), init='normal', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(num_of_features / 4, init='normal', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(len(CLASSES), init='normal', activation='softmax'))
    # # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', fbeta_custom])
    return model


def five_layer_nn(num_of_features=0):
    K.set_learning_phase(1)

    # # create model
    model = Sequential()
    model.add(Dense(
        num_of_features,
        input_dim=num_of_features,
        W_regularizer=l1l2(0.01),
        init='normal',
        activation='relu'
    ))
    # model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Dense(int(num_of_features * 1.5), init='normal', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(num_of_features / 2, init='normal', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(num_of_features / 4, init='normal', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(len(CLASSES), init='normal', activation='softmax'))

    # # define loss optimizer
    adam = Adam(lr=0.0002)

    # # compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', fbeta_custom])
    return model


model_dict = {
    'three_layer_nn': three_layer_nn,
    'four_layer_nn': four_layer_nn,
    'five_layer_nn': five_layer_nn,
}


def preprocess(
    df,
    trial,
):
    # # shuffle dataframe rows
    if trial:
        frac = 0.25
    else:
        frac = 1
    df = df.sample(frac=frac).reset_index(drop=True)
    dataset = df.values

    # # split into X and Y
    num_of_features = dataset.shape[1] - 1
    X = dataset[:, 0:num_of_features]
    Y = dataset[:, num_of_features]

    # # bin data into categories
    bins = CLASS_BINS
    bin_names = CLASSES
    categories = pd.cut(Y, bins, labels=bin_names)

    # # one hot encode
    Y = pd.get_dummies(categories).values

    # # split into training, test and validation sets
    num_of_samples = X.shape[0]
    val_ratio = 0.2
    test_ratio = 0.1
    train_ratio = 1 - val_ratio - test_ratio
    num_of_val_samples = int(val_ratio * num_of_samples)
    num_of_train_samples = int(train_ratio * num_of_samples)

    X_train = X[0:(num_of_train_samples + 1), :]
    Y_train = Y[0:(num_of_train_samples + 1), :]
    X_val = X[num_of_train_samples:(num_of_train_samples + num_of_val_samples + 1), :]
    Y_val = Y[num_of_train_samples:(num_of_train_samples + num_of_val_samples + 1), :]
    X_test = X[num_of_train_samples + num_of_val_samples:, :]
    Y_test = Y[num_of_train_samples + num_of_val_samples:, :]

    # # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # # standardize data and store mean and scale to file
    scaler = StandardScaler().fit(X_train)
    mean_array = scaler.mean_
    scale_array = scaler.scale_
    X_train_transformed = scaler.transform(X_train)
    X_val_transformed = (X_val - mean_array) / (scale_array)
    X_test_transformed = (X_test - mean_array) / (scale_array)

    # # return data
    data_dict = {
        'train_data': (X_train_transformed, Y_train),
        'val_data': (X_val_transformed, Y_val),
        'test_data': (X_test_transformed, Y_test),
        'norm_arrays': (mean_array, scale_array),
        'num_of_features': num_of_features,
    }

    return data_dict


def save_model_files(
    model_name,
    model,
    position,
    mean_array,
    scale_array
):
    mean_list = [mean for mean in mean_array]
    scale_list = [scale for scale in scale_array]
    # # dump model files
    model_filepath = 'dumps/%s/keras_%ss/keras_%ss.json' % (model_name, position, position)
    create_filepath(model_filepath)
    weights_filepath = 'dumps/%s/keras_%ss/weights.h5' % (model_name, position)
    create_filepath(weights_filepath)
    mean_filepath = 'dumps/%s/keras_%ss/mean.json' % (model_name, position)
    create_filepath(mean_filepath)
    scale_filepath = 'dumps/%s/keras_%ss/scale.json' % (model_name, position)
    create_filepath(scale_filepath)

    # model.save(model_filepath)
    model.save_weights(weights_filepath)
    model_json = model.to_json()

    with open(model_filepath, "w+") as f:
        f.write(model_json)
    with open(mean_filepath, 'w+') as f:
        f.write(json.dumps(mean_list))
    with open(scale_filepath, 'w+') as f:
        f.write(json.dumps(scale_list))


def train(
    position='forward',
    model_name='three_layer_nn',
    use_class_weights=False,
    smote=False,
    undersampling=False,
    trial=False,
):
    """Creates a network, trains it and dumps it on disk
    """

    # # load dataset
    data_path = 'fpl_%ss.ssv' % (position)
    df = pd.read_csv(data_path, delim_whitespace=True, header=None)

    # # preprocess data
    processed_data = preprocess(df=df, trial=trial)
    X_train_transformed, Y_train = processed_data['train_data']
    X_val_transformed, Y_val = processed_data['val_data']
    X_test_transformed, Y_test = processed_data['test_data']
    mean_array, scale_array = processed_data['norm_arrays']
    num_of_features = processed_data['num_of_features']

    X_train_resampled, Y_train_resampled = X_train_transformed, Y_train

    smote_ratio = 1.0
    if undersampling:
        # # Apply random undersampling for dominant class
        X_train_resampled, Y_train_resampled = apply_undersampling(X_train_resampled, Y_train_resampled)
    if smote:
        # # Apply regular SMOTE
        smote_ratio = 0.5
        X_train_resampled, Y_train_resampled = apply_smote(X_train_resampled, Y_train_resampled, ratio=smote_ratio)

    # # evaluate model with standardized dataset
    CLASS_WEIGHT_UNIFORM = {
        0: 1.,
        1: 1.0,
        2: 1.0,
    }
    if use_class_weights:
        class_weight_dict = CLASS_WEIGHT[position]
    else:
        class_weight_dict = CLASS_WEIGHT_UNIFORM
    print('Class weights - ')
    print(class_weight_dict)
    model_fn = model_dict[model_name]
    model = model_fn(num_of_features=num_of_features)
    # print(Y_train_resampled[0:10, :])
    history = model.fit(
        X_train_resampled,
        Y_train_resampled,
        nb_epoch=50,
        batch_size=10,
        verbose=0,
        # validation_split=0.2,
        validation_data=(X_val_transformed, Y_val),
        class_weight=class_weight_dict,
    )
    val_predictions_keras = model.predict(X_val_transformed)
    test_predictions_keras = model.predict(X_test_transformed)
    # print(test_predictions_keras[0:10])

    # # generate and print confusion matrix for validation and test data
    val_conf_matrix = get_confusion_matrix_one_hot(val_predictions_keras, Y_val)
    print('validation confusion matrix - ')
    print(val_conf_matrix)
    conf_matrix = get_confusion_matrix_one_hot(test_predictions_keras, Y_test)
    print('test confusion matrix - ')
    print(conf_matrix)

    if not trial:
        model_name = model_fn.__name__
        save_model_files(model_name, model, position, mean_array, scale_array)

    # plot training history

    # list all data in history
    print(history.history.keys())

    # print final metrics
    print('final train loss = %s' % history.history['loss'][-1])
    print('final validation loss = %s' % history.history['val_loss'][-1])
    print('final train fbeta = %s' % history.history['fbeta_custom'][-1])
    print('final validation fbeta = %s' % history.history['val_fbeta_custom'][-1])

    # summarize history for accuracy
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for f score
    plt.clf()
    plt.plot(history.history['fbeta_custom'])
    plt.plot(history.history['val_fbeta_custom'])
    plt.title('model f score')
    plt.ylabel('f score')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
