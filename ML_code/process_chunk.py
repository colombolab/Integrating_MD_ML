import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


def computing_metrics(confusion_matrix):
    confusion_matrix = np.asarray(confusion_matrix)
    tp = confusion_matrix[:, 0, 0]
    fp = confusion_matrix[:, 0, 1]
    fn = confusion_matrix[:, 1, 0]
    tn = confusion_matrix[:, 1, 1]
    acc_divisor = tp + tn + fp + fn
    accuracy = np.divide(tp + tn, acc_divisor, out=np.zeros_like(tp, dtype=float), where=(acc_divisor != 0))
    tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    tnr = np.divide(tn, tn + fp, out=np.zeros_like(tp, dtype=float), where=(tn + fp) != 0)
    ppv = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    mcc_divisor = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, mcc_divisor, out=np.zeros_like(tp, dtype=float), where=(mcc_divisor != 0))
    return accuracy, tpr, tnr, ppv, f1, mcc


def evaluation_metrics(ytest, prediction):
    confusion_matrix = multilabel_confusion_matrix(ytest, prediction, labels=[0, 1])
    accuracy, tpr, tnr, ppv, f1, mcc = computing_metrics(confusion_matrix)
    return np.array([np.mean(accuracy), np.mean(tpr), np.mean(tnr), np.mean(ppv), np.mean(f1), np.mean(mcc)])


def process_chunk(chunk, xdata, ydata, model, probability):
    train_index, test_index = chunk
    xtrain, ytrain = xdata[train_index], ydata[train_index]
    xtest, ytest = xdata[test_index], ydata[test_index]
    model.fit(xtrain, ytrain)
    proba = model.predict_proba(xtest)
    diff = np.abs(proba[:, 0] - proba[:, 1])
    condition_1 = (proba[:, 0] > proba[:, 1]) & (diff >= 2 * probability - 1)
    condition_2 = (proba[:, 1] > proba[:, 0]) & (diff >= 2 * probability - 1)
    prediction = np.where(condition_1, 0, np.where(condition_2, 1, 2))
    return evaluation_metrics(ytest, prediction)
