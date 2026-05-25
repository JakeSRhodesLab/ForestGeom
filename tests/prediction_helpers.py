import numpy as np


def normalize_rows(P):
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    return P.multiply((1.0 / row_sums)[:, None]).tocsr()


def predict_regression_from_proximity(P, y_train):
    return np.asarray(P @ np.asarray(y_train)).ravel()


def predict_classifier_from_proximity(P, y_train, classes):
    y_train = np.asarray(y_train)
    classes = np.asarray(classes)
    proba = np.zeros((P.shape[0], classes.shape[0]), dtype=np.float64)
    for class_index, cls in enumerate(classes):
        proba[:, class_index] = np.asarray(P[:, y_train == cls].sum(axis=1)).ravel()

    preds = classes[np.argmax(proba, axis=1)]
    return preds, proba