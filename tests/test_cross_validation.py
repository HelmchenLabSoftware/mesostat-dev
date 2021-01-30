import numpy as np
import matplotlib.pyplot as plt
from mesostat.stat.classification import confusion_matrix, weighted_accuracy, leave_one_out_iterator

'''
* Have completely random data
* apply a leave-one-out iterator
* Construct a confusion matrix on test set
* Compute weighted accuracy

Expected result should be close to 50% accuracy. Previously, there was a bug in confusion_matrix routine which is caught
by this test
'''

nStuff = 100

acc = []
for i in range(100):

    x = np.random.randint(0, 2, nStuff)
    y = np.random.randint(0, 2, nStuff)
    it = leave_one_out_iterator(x, y)
    cm = np.zeros((2, 2))
    for xTrain, yTrain, xTest, yTest in it:
        cm += confusion_matrix(xTest, yTest, [0, 1])
    acc += [weighted_accuracy(cm)]

print(np.mean(acc))

plt.figure()
plt.plot(acc)
plt.show()

