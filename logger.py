
import numpy as np
import statistics


from sklearn.model_selection import train_test_split
def stat(acc_list, metrics):
    mean = statistics.mean(acc_list)
    stdev = statistics.stdev(acc_list)
    print(f'Final {metrics}  using 10 fold CV: {mean * 100:.2f} \u00B1 {stdev * 100:.2f}%')
    return mean*100, stdev*100


def print_stat(train_acc, test_acc):
    argmax = np.argmax(train_acc)
    best_result = test_acc[argmax]
    train_acc = np.max(train_acc)
    test_acc = np.max(test_acc)
    print(f'Train accuracy = {train_acc:.4f}%,Test Accuracy = {test_acc:.4f}%\n')
    return test_acc