"""
This module has the functions for training and testing the model,
and model inference
"""
from ml.data import process_data
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import sys
print(sys.path)

sys.path.append("./")
sys.path.append("./src")


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = AdaBoostClassifier(n_estimators=60, random_state=42)
    model.fit(X_train, y_train)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    cv_report = cross_val_score(model, X_train, y_train,
                                cv=cv, n_jobs=-1)

    print('KFold CV report.')
    print(cv_report)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)

    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : pkl
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    return model.predict(X)


def slices_performances(data, model, encoder, lb, cat_features):
    """Check performance model metrics
    on slices of categorical features"""

    slice_file = open('model/slice_output.txt', 'w')

    print('running performance metrics in slices')
    for feature in cat_features:
        for current_class in data[feature].unique():
            df_current_class = data[data[feature] == current_class]

            X_test, y_test, _, _ = process_data(
                df_current_class,
                cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False)

            y_pred = model.predict(X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            row = f"{feature} and current_class={current_class}"\
                f" - Precision: {precision: .2f}."\
                f" Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
            print(row)
            slice_file.write(row + '\n')

    print("Performance metrics for slices: saved in model/slice_output.txt")
