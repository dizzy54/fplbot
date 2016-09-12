from lib.create_dataset import load_dataset

from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_validation import KFold
# from sklearn.metrics import mean_squared_error
import numpy as np

midfielder_data = load_dataset(position='goalkeeper')
X = np.array(midfielder_data[0])
Y = np.array(midfielder_data[1])

# Y[Y > 10.0] = 10.0
# num_of_features = X.shape[1]

num_of_samples = X.shape[0]
test_ratio = 0.1
num_of_test_samples = int(test_ratio * num_of_samples)
num_of_train_samples = num_of_samples - num_of_test_samples

X_train = X[0:(num_of_train_samples + 1), :]
Y_train = Y[0:(num_of_train_samples + 1)]
X_test = X[num_of_train_samples:, :]
Y_test = Y[num_of_train_samples:]

l1_ratio = [0.01, .05, 0.25, 0.5, 0.75, .95, 0.99]
met = ElasticNetCV(l1_ratio=l1_ratio, n_jobs=-1, normalize=True)

kf = KFold(len(Y_train), n_folds=5)
pred = np.zeros_like(Y_train)

for train, test in kf:
    met.fit(X_train[train], Y_train[train])
    pred[test] = met.predict(X_train[test])

print met.score(X_train, Y_train)

test_predictions = met.predict(X_test)

prediction_error = abs(test_predictions - Y_test)
# print prediction_error
cutoff_error = 0.4 * Y_test
cutoff_error[cutoff_error < 2] = 2
prediction_success = prediction_error < cutoff_error

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

# print prediction_success
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

'''
# # scatter plot
plt.scatter(test_predictions, Y_test)
plt.plot([test_predictions.min(), test_predictions.max()], [test_predictions.min(), test_predictions.max()])
plt.show()

plt.clf()
'''

# # frequency plot
plt.plot(y_values, y_freq)
plt.plot(y_values, y_pred_correct_freq)
plt.show()
