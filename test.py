from lib.create_dataset import load_dataset

from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_validation import KFold
# from sklearn.metrics import mean_squared_error
import numpy as np

midfielder_data = load_dataset(position='midfielder')
data = np.array(midfielder_data[0])
target = np.array(midfielder_data[1])

l1_ratio = [0.01, .05, 0.25, 0.5, 0.75, .95, 0.99]
met = ElasticNetCV(l1_ratio=l1_ratio, n_jobs=-1, normalize=True)

kf = KFold(len(target), n_folds=5)
pred = np.zeros_like(target)

for train, test in kf:
    met.fit(data[train], target[train])
    pred[test] = met.predict(data[test])

print met.score(data, target)

plt.scatter(pred, target)
plt.plot([pred.min(), pred.max()], [pred.min(), pred.max()])
plt.show()
