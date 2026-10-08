import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from api import app  # noqa: E402


@pytest.fixture(scope='module')
def client():
	with TestClient(app) as c:
		yield c


def test_health(client):
	response = client.get('/health')
	assert response.status_code == 200
	assert response.json()['status'] == 'ok'


def test_ingredients(client):
	response = client.get('/ingredients')
	assert response.status_code == 200
	assert 'milk' in response.json()


def test_predict(client):
	response = client.post('/predict', json={'ingredients': ['Milk', ' honey', 'jam']})
	assert response.status_code == 200
	body = response.json()
	assert body['rating_class'] in {'bad', 'so-so', 'great'}
	assert set(body['probabilities']) == {'bad', 'so-so', 'great'}
	assert sum(body['probabilities'].values()) == pytest.approx(1.0)
	assert body['rating_class'] == max(body['probabilities'], key=body['probabilities'].get)


def test_predict_unknown_ingredient(client):
	response = client.post('/predict', json={'ingredients': ['milk', 'unobtainium']})
	assert response.status_code == 422
	assert response.json()['detail']['unknown'] == ['unobtainium']


def test_predict_empty(client):
	assert client.post('/predict', json={'ingredients': []}).status_code == 422
	assert client.post('/predict', json={'ingredients': ['  ']}).status_code == 422


def test_nutrition(client):
	response = client.post('/nutrition', json={'ingredients': ['almond']})
	assert response.status_code == 200
	assert response.json()['almond']['Protein'] > 0
