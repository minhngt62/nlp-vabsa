import numpy as np

def f1(y_pred, y):
    final_f1 = 0
    eps = 1e-7
    for i in range(len(y)):
        precision = np.sum(np.where(y_pred[i] == y[i], y[i] != 0, False)) / (np.sum(np.where(y_pred[i] != 0, 1, 0)) + eps)
        recall = np.sum(np.where(y_pred[i] == y[i], y[i] != 0, False)) / (np.sum(np.where(y[i] != 0, 1, 0)) + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        final_f1 += f1
    return final_f1 / len(y_pred)
        
