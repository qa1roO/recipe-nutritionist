"""
HTTP API for the recipe rating classifier.

The model comes from the MLflow Model Registry when MODEL_URI is set
(e.g. models:/recipe-rating-classifier@champion), otherwise from src/best_model.pkl.

Run locally:
	uvicorn api:app --app-dir src --port 8000
"""

import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MODEL_PATH, NUTRIENTS_FACTS_PATH


class Ingredients(BaseModel):
	ingredients: list[str] = Field(min_length=1, examples=[['milk', 'honey', 'jam']])


class Prediction(BaseModel):
	rating_class: str
	probabilities: dict[str, float]


state = {}


def load_model():
	model_uri = os.getenv('MODEL_URI')
	if model_uri:
		import mlflow.sklearn
		return mlflow.sklearn.load_model(model_uri), model_uri
	return joblib.load(MODEL_PATH), str(MODEL_PATH.name)


def feature_names(model) -> list[str]:
	names = getattr(model, 'feature_names_', None)
	if names is None:
		names = getattr(model, 'feature_names_in_')
	return list(names)


@asynccontextmanager
async def lifespan(app: FastAPI):
	model, source = load_model()
	state['model'] = model
	state['model_source'] = source
	state['features'] = feature_names(model)
	state['nutrients'] = pd.read_csv(NUTRIENTS_FACTS_PATH, index_col=0)
	yield
	state.clear()


app = FastAPI(
	title='Recipe Nutritionist API',
	description='Predicts the rating class of a dish by its ingredients and shows their nutrition facts.',
	version='0.2.0',
	lifespan=lifespan,
)


def normalize(ingredients: list[str], known: list[str]) -> list[str]:
	cleaned = sorted({i.strip().lower() for i in ingredients if i.strip()})
	unknown = [i for i in cleaned if i not in known]
	if not cleaned or unknown:
		raise HTTPException(
			status_code=422,
			detail={'message': 'Unknown ingredients, see GET /ingredients', 'unknown': unknown},
		)
	return cleaned


@app.get('/health')
def health():
	return {'status': 'ok', 'model': state['model_source']}


@app.get('/ingredients')
def ingredients() -> list[str]:
	return state['features']


@app.post('/predict')
def predict(request: Ingredients) -> Prediction:
	features = state['features']
	selected = normalize(request.ingredients, features)
	vector = pd.DataFrame([[0] * len(features)], columns=features)
	vector[selected] = 1

	model = state['model']
	probabilities = model.predict_proba(vector)[0]
	classes = [str(c) for c in model.classes_]
	scores = dict(zip(classes, map(float, probabilities)))
	return Prediction(rating_class=max(scores, key=scores.get), probabilities=scores)


@app.post('/nutrition')
def nutrition(request: Ingredients) -> dict[str, dict[str, float]]:
	"""Percent of the daily value per nutrient for each ingredient (non-zero values only)."""
	facts = state['nutrients']
	selected = normalize(request.ingredients, list(facts.columns))
	return {
		ingredient: {k: round(float(v), 2) for k, v in facts[ingredient].items() if v != 0}
		for ingredient in selected
	}
