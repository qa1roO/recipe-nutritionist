#!/usr/bin/env python
"""
Reproducible training of the rating classifier with experiment tracking in MLflow.

Every candidate model is a separate MLflow run with its best hyperparameters,
cross-validation score and test metrics. The best model by precision of the
'great' class is logged with its signature, registered in the Model Registry
and exported to src/best_model.pkl for the CLI and the API.

Usage:
	python src/train.py
	MLFLOW_TRACKING_URI=http://localhost:5000 python src/train.py
"""

import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import (
	BLACKLIST,
	MODEL_PATH,
	RANDOM_STATE,
	RECIPES_PATH,
	REGISTERED_MODEL_NAME,
	TARGET_CLASS,
)
from recipes import RecipeDataPreprocessor

EXPERIMENT_NAME = 'recipe-rating'


def rating_to_class(rating: float) -> str:
	rounded = int(rating + 0.5)
	if rounded <= 1:
		return 'bad'
	if rounded <= 3:
		return 'so-so'
	return 'great'


def load_dataset():
	source_df = pd.read_csv(RECIPES_PATH, encoding='utf-8')
	df = RecipeDataPreprocessor(blacklist=BLACKLIST).fit_transform(source_df)
	X = df.drop(columns='rating')
	y = df['rating'].apply(rating_to_class)
	return X, y


def target_precision(y_true, y_pred):
	return precision_score(y_true == TARGET_CLASS, y_pred == TARGET_CLASS, zero_division=0)


def candidate_models():
	return {
		'DecisionTree': (
			DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
			{
				'criterion': ['gini'],
				'max_depth': [5, 10, None],
				'min_samples_split': [5, 10],
				'min_samples_leaf': [1, 4],
			},
		),
		'RandomForest': (
			RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1),
			{
				'n_estimators': [100],
				'max_depth': [10, None],
				'min_samples_leaf': [1, 4],
			},
		),
		'CatBoost': (
			CatBoostClassifier(
				loss_function='MultiClass',
				auto_class_weights='Balanced',
				random_seed=RANDOM_STATE,
				thread_count=1,
				verbose=False,
				allow_writing_files=False,
			),
			{
				'iterations': [150],
				'depth': [4, 6],
				'learning_rate': [0.05, 0.1],
			},
		),
	}


def test_metrics(model, X_test, y_test):
	y_pred = model.predict(X_test)
	# CatBoost returns a column vector of labels
	y_pred = pd.Series(pd.DataFrame(y_pred).iloc[:, 0].values, index=y_test.index)
	return {
		'test_precision_great': target_precision(y_test, y_pred),
		'test_accuracy': accuracy_score(y_test, y_pred),
		'test_f1_macro': f1_score(y_test, y_pred, average='macro'),
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--cv', type=int, default=5, help='number of cross-validation folds')
	parser.add_argument('--n-jobs', type=int, default=-1, help='parallel jobs for grid search')
	parser.add_argument('--no-export', action='store_true', help=f'do not overwrite {MODEL_PATH.name}')
	args = parser.parse_args()

	mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
	mlflow.set_experiment(EXPERIMENT_NAME)

	X, y = load_dataset()
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
	)
	dataset_params = {
		'n_recipes': len(X),
		'n_features': X.shape[1],
		'test_size': 0.2,
		'cv_folds': args.cv,
		'selection_metric': f'precision_{TARGET_CLASS}',
	}

	with mlflow.start_run(run_name='baseline-most-frequent'):
		baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
		mlflow.log_params(dataset_params)
		mlflow.log_metrics(test_metrics(baseline, X_test, y_test))

	scorer = make_scorer(target_precision)
	results = {}
	for name, (estimator, grid) in candidate_models().items():
		with mlflow.start_run(run_name=name) as run:
			search = GridSearchCV(estimator, grid, scoring=scorer, cv=args.cv, n_jobs=args.n_jobs)
			search.fit(X_train, y_train)
			metrics = {'cv_precision_great': search.best_score_, **test_metrics(search.best_estimator_, X_test, y_test)}
			mlflow.log_params({'model': name, **dataset_params, **search.best_params_})
			mlflow.log_metrics(metrics)
			results[name] = (search.best_estimator_, metrics, run.info.run_id)
			print(f'{name}: {search.best_params_} ' + ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

	# Choose by cross-validation, never by the test set
	best_name = max(results, key=lambda n: results[n][1]['cv_precision_great'])
	best_model, best_metrics, run_id = results[best_name]

	with mlflow.start_run(run_id=run_id):
		example = X_test.head(5)
		model_info = mlflow.sklearn.log_model(
			best_model,
			name='model',
			signature=infer_signature(example, best_model.predict(example)),
			input_example=example,
			registered_model_name=REGISTERED_MODEL_NAME,
			# skops refuses classes outside scikit-learn (e.g. CatBoost) unless they are trusted explicitly
			skops_trusted_types=[f'{type(best_model).__module__}.{type(best_model).__qualname__}'],
		)
	client = MlflowClient()
	version = model_info.registered_model_version
	client.set_registered_model_alias(REGISTERED_MODEL_NAME, 'champion', version)
	print(f'\nBest: {best_name}, registered as {REGISTERED_MODEL_NAME} v{version} (alias "champion")')
	print(', '.join(f'{k}={v:.4f}' for k, v in best_metrics.items()))

	if not args.no_export:
		joblib.dump(best_model, MODEL_PATH)
		print(f'Exported to {MODEL_PATH}')


if __name__ == '__main__':
	main()
