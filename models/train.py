import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id=104434781469597209)

#load the CSV file
dir = Path(__file__).parent.parent / "data/Crop_recommendation.csv"
df = pd.read_csv(
    dir,
    encoding='utf-8'
)

#list all the feature columns
df_features = df.columns.to_list()[:-1]

#split into train and test
X = df[df_features].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

#encode labels to numeric values
prc = LabelEncoder()
y_train = prc.fit_transform(y_train)

with mlflow.start_run():

    mlflow.sklearn.autolog()

    #set up pipeline for logistic regression
    pipe_lr = Pipeline(
        [('prc', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', penalty='l2', C=10.))]
    )

    pipe_lr.fit(X_train, y_train)

    #calculate accuracy score
    y_test = prc.transform(y_test)
    acc_train = pipe_lr.score(X_train, y_train)
    acc_test = pipe_lr.score(X_test, y_test)

    mlflow.log_metrics(
        {"acc_train": acc_train,
         "acc_test": acc_test}
    )

    print(f'Train accuracy: {acc_train:.1%}')
    print(f'Test accuracy: {acc_test:.1%}')