import numpy as np

def multioutput_f1(y, y_pred, f1_only=True):
    finale = {"f1": 0, "precision": 0, "recall": 0}
    eps = 1e-7
    for i in range(len(y)):
        precision = np.sum(np.where(y_pred[i] == y[i], y[i] != 0, False)) / (np.sum(np.where(y_pred[i] != 0, 1, 0)) + eps)
        recall = np.sum(np.where(y_pred[i] == y[i], y[i] != 0, False)) / (np.sum(np.where(y[i] != 0, 1, 0)) + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        finale["f1"] += f1 / len(y_pred)
        finale["precision"] += precision / len(y_pred)
        finale["recall"] += recall / len(y_pred)
    if f1_only:
        return finale["f1"]
    return finale
        
