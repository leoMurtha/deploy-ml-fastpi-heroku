Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up

* Download and install conda if you don’t have it already.
  * Use the supplied requirements file to create a new environment, or
  * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
  * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Train model

To train and the test the model run: `python src/train_model`


## Tests

To test the pipeline: `pytest`


## Tests API on Heroku

To test the API: `python heroku_live_api.py`
