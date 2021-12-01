# Importing Packages
import glob2
import pandas as pd
import logging
from Preprocessing.data_loader import Dataload
from Preprocessing.preprocessing import DataPreProcessor
from Regression_Models.Regression import LR_With_FeatureSelection, Embedded_method_for_feature_selection
from Regression_Models.tree_models import Tree_models_regression
from Regression_Models.Ensemble_models import Ensemble_models
import glob
import joblib
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)

# Source of dataset
dataset = glob2.glob("static/files/*.csv")

# configuring logging operation
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')

# Loading the dataset feeding it as string here is a list due to glob function
d = Dataload(dataset[0])

# Output is nothing but the data in the form of pandas dataframe
df = d.fetch_data()

# Data Preprocessing
dp = DataPreProcessor(df)

# Splitting the data into train & test sets resp.
df_train, df_test = dp.data_split(test_size=0.30)

# Feature Scaling for linear regression
df_train_scale, df_test_scale = dp.feature_scaling(df_train, df_test)

# Splitting dataset into independent and dependent dataset resp.
X_train, y_train, X_test, y_test = dp.train_test_splitting(df_train, df_test, 'Age')