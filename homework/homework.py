import pandas as pd
import numpy as np
import gzip
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix, make_scorer
import json
import os

# Cargar datos
train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

# Preprocesamiento de datos
train_data.rename(columns={'default payment next month': 'default'}, inplace=True)
test_data.rename(columns={'default payment next month': 'default'}, inplace=True)
train_data.drop(columns=['ID'], inplace=True, errors='ignore')
test_data.drop(columns=['ID'], inplace=True, errors='ignore')
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
train_data = train_data[(train_data["EDUCATION"] != 0) & (train_data["MARRIAGE"] != 0)]
test_data = test_data[(test_data["EDUCATION"] != 0) & (test_data["MARRIAGE"] != 0)]
train_data["EDUCATION"] = train_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
test_data["EDUCATION"] = test_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

# Separar características y etiquetas
x_train, y_train = train_data.drop(columns=['default']), train_data['default']
x_test, y_test = test_data.drop(columns=['default']), test_data['default']
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features = list(set(x_train.columns) - set(categorical_features))

# Crear pipeline
transformer = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('scaler', StandardScaler(), numerical_features)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', transformer),
    ('pca', PCA()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=12)),
    ('classifier', SVC(kernel="rbf", random_state=12345, max_iter=-1))
])

# Ajustar hiperparámetros
params = {
    "pca__n_components": [20, x_train.shape[1] - 2],
    'feature_selection__k': [12],
    'classifier__kernel': ["rbf"],
    'classifier__gamma': [0.1]
}

cv = StratifiedKFold(n_splits=10)
scorer = make_scorer(balanced_accuracy_score)
grid_search = GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=cv, n_jobs=-1, refit=True)
grid_search.fit(x_train, y_train)

# Guardar modelo
os.makedirs("files/models/", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", 'wb') as f:
    pickle.dump(grid_search, f)

# Calcular métricas
def compute_metrics(y_true, y_pred, dataset):
    return {
        'type': 'metrics',
        'dataset': dataset,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

def confusion_matrix_data(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': dataset,
        'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

metrics = [
    compute_metrics(y_train, grid_search.predict(x_train), 'train'),
    compute_metrics(y_test, grid_search.predict(x_test), 'test'),
    confusion_matrix_data(y_train, grid_search.predict(x_train), 'train'),
    confusion_matrix_data(y_test, grid_search.predict(x_test), 'test')
]

# Guardar métricas
os.makedirs("files/output/", exist_ok=True)
with open("files/output/metrics.json", "w") as f:
    for metric in metrics:
        f.write(json.dumps(metric) + "\n")