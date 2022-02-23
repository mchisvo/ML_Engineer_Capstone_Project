from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
from sklearn.externals import joblib
# Import joblib package directly
#import joblib

# import sklearn library
from sklearn.ensemble import RandomForestClassifier



# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()
    
    
    # hyperparameters sent by the client
    parser.add_argument("--bootstrap", type=bool, default=True)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--max_features", type=str, default='auto')
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--n_estimators", type=int, default=10)
    
    
    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    
    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    
    # Labels are in the first column
    y_train = train_data.iloc[:,0]
    X_train = train_data.iloc[:,1:]
    
    # train The RF classifier
    print("training model")
    model = RandomForestClassifier(
        bootstrap=args.bootstrap, 
        max_depth=args.max_depth, 
        max_features=args.max_features, 
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        n_estimators=args.n_estimators,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
