# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Using the AdaBoost classifier from Sklearn with simple hyperparameters.

## Intended Use

Model to infer a person might exceed a income of $50K/yr based on census data.

## Training Data

The data cointains information from the 1994 Census database.

## Evaluation Data

Using a 80-20 split was used to break this into a train and test set. Pre-processing encoders are only created using training data to avoid data leakage.

## Metrics

Metrics on test set: 

Precision: 0.69; 

Recall: 0.68.

Fbeta: 0.69.

## Ethical Considerations

Given that the data contains attributes about sex, race and so on, special consideration should be given to how the model
performs accros different groups.

## Caveats and Recommendations

To further improve the performance:

* Use GridSearch or RandomSearch hyperparameter optimization.
* Invest more time in feature engineering.
