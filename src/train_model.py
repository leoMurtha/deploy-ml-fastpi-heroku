# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import cat_features, process_data, load_data
import argparse
from ml.utils import save_object
from ml.model import train_model, compute_model_metrics
from ml.model import inference, slices_performances

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--model-output', type=str, required=True)
    parser.add_argument('--encoder-output', type=str, required=True)
    parser.add_argument('--lb-output', type=str, required=True)

    # Parse the argument
    args = parser.parse_args()

    data = load_data(args.data_path)

    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    y_pred = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    print('Test set/Holdout Set metrics: \n')
    print(f'precision: {precision}')
    print(f'\nrecall: {recall}')
    print(f'fbeta: {fbeta}\n')

    # slices on the test set
    slices_performances(test, model, encoder, lb, cat_features)

    # Saving objects
    save_object(model, args.model_output)
    save_object(encoder, args.encoder_output)
    save_object(lb, args.lb_output)
