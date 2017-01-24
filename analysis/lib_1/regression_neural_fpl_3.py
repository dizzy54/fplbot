# import random
import numpy as np
import pandas
import json
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.regularizers import l1l2
# from keras.wrappers.scikit_learn import KerasRegressor
# from keras.layers.normalization import BatchNormalization
# from sklearn.cross_validation import cross_val_score
# from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
# from sklearn.externals import joblib

import sys
import os
import errno

# trial = True


# # to create dir if dir does not exist
def create_filepath(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# # define base model
def baseline_model(num_of_features=0):
    # # create model
    K.set_learning_phase(1)
    model = Sequential()
    model.add(Dense(
        num_of_features,
        input_dim=num_of_features,
        W_regularizer=l1l2(0.3),
        init='normal',
        activation='relu'
    ))
    # model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(int(num_of_features * 1.5), init='normal', activation='relu', W_constraint=maxnorm(3)))
    # model.add(BatchNormalization())
    model.add(Dense(num_of_features / 4, init='normal', activation='relu', W_constraint=maxnorm(3)))
    # model.add(BatchNormalization())
    model.add(Dense(1, init='normal'))
    # # compile model
    model.compile(loss='mean_squared_error', optimizer='adam', )
    return model


def create_model_dump(position='midfielder', model_fn=baseline_model, trial=True):
    """Creates a vanilla neural network and dumps it on disk
    """
    sys.setrecursionlimit(10000)

    # # load dataset
    data_path = 'fpl_%ss.ssv' % (position)
    df = pandas.read_csv(data_path, delim_whitespace=True, header=None)
    # # shuffle dataframe rows
    if trial:
        frac = 0.1
    else:
        frac = 1
    df = df.sample(frac=frac).reset_index(drop=True)
    dataset = df.values
    # # split into X and Y
    num_of_features = dataset.shape[1] - 1
    X = dataset[:, 0:num_of_features]
    Y = dataset[:, num_of_features]

    # Y[Y > 10.0] = 10.0
    num_of_samples = X.shape[0]
    test_ratio = 0.1
    num_of_test_samples = int(test_ratio * num_of_samples)
    num_of_train_samples = num_of_samples - num_of_test_samples

    X_train = X[0:(num_of_train_samples + 1), :]
    Y_train = Y[0:(num_of_train_samples + 1)]
    X_test = X[num_of_train_samples:, :]
    Y_test = Y[num_of_train_samples:]

    # # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # # standardize data and store mean and scale to file
    scaler = StandardScaler().fit(X_train)
    mean_array = scaler.mean_
    scale_array = scaler.scale_
    X_train_transformed = scaler.transform(X_train)

    mean_list = [mean for mean in mean_array]
    scale_list = [scale for scale in scale_array]

    # # evaluate model with standardized dataset
    model = model_fn(num_of_features=num_of_features)
    history = model.fit(X_train_transformed, Y_train, nb_epoch=300, batch_size=10, verbose=0, validation_split=0.2)
    X_test_transformed = (X_test - mean_array) / (scale_array)
    test_predictions_keras = model.predict(X_test_transformed)
    test_predictions = [pred[0] for pred in test_predictions_keras]

    model_name = model_fn.__name__

    if not trial:
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

    # plot training history

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    '''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # # print test_predictions

    prediction_error = abs(test_predictions - Y_test)
    print 'max_prediction = %s' % max(test_predictions)
    # print prediction_error
    cutoff_error = 0.4 * Y_test
    cutoff_error[cutoff_error < 2] = 2
    prediction_success = prediction_error < cutoff_error
    # print prediction_success

    y_values = range(0, 11, 1)
    y_freq = []
    y_pred_correct_freq = []
    for y in y_values:
        y_this = np.copy(Y_test)
        y_this[y_this != y] = -10
        y_this[y_this != -10] = 1
        y_this[y_this == -10] = 0
        y_num = np.sum(y_this)
        y_freq.append(y_num)
        y_pred_correct = np.sum(np.multiply(y_this, prediction_success))
        print "for y = %s, n = %s, correct = %s" % (y, y_num, y_pred_correct)
        y_pred_correct_freq.append(y_pred_correct)
        # print y_num
        # print y_pred_correct

    accuracy = float(np.sum(prediction_success)) / num_of_test_samples * 100
    print 'Accuracy on test data is %s percent on training data of %s rows, test data of %s rows' % (
        accuracy,
        num_of_train_samples,
        num_of_test_samples
    )

    high_cutoff = 4
    Y_correct_freq_high = [n_correct for n_correct in y_pred_correct_freq[high_cutoff:]]
    Y_freq_high = [n for n in y_freq[high_cutoff:]]
    accuracy = float(sum(Y_correct_freq_high)) / sum(Y_freq_high) * 100
    print 'Accuracy on test data is %s percent for high y values' % (
        accuracy,
    )
    plt.clf()

    # # frequency plot
    plt.plot(y_values, y_freq)
    plt.plot(y_values, y_pred_correct_freq)
    plt.show()

    plt.clf()
    plt.scatter(test_predictions, Y_test)
    plt.show()
