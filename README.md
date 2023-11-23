# Recommendation System Pipeline

This recommendation system pipeline is designed to predict movie ratings given by users. It consists of several Python files that handle different aspects of the recommendation process.

## Files Overview

### main.py

- **get_args():** Parses command-line arguments to determine the mode (train or test) and file paths.
- **Main block:** Executes either the training or testing phase based on the specified mode.

### train_model.py

- **Functionality:** Handles the training of machine learning models for the recommendation system.
- **Steps:**
  - Preprocessing
  - Feature engineering
  - Model training
  - Evaluation and comparison

### model_training.py

- **Functionality:** Trains various machine learning models, including XGBoost, SVD/SVD++, and Surprise modules.

### data_featurizing.py

- **Functionality:** Featurizes data for user-user and movie-movie similarities using Cosine Similarity values.

### utility.py

- **Functionality:** Utility functions for loading various functions utilized in other scripts.

### test_model.py

- **Functionality:** Tests the trained models on new data.
- **Steps:**
  - Preprocessing
  - Prediction
  - Evaluation and comparison

## How to Run

### Train Model

```bash
python main.py --mode=train --train_file=<training file path>
