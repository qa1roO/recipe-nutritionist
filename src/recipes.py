import re
import os
import numpy as np
import pandas as pd
import joblib
try:
	from tqdm.auto import tqdm
except ModuleNotFoundError:
	class tqdm:
		def __init__(self, iterable=None, **kwargs):
			self.iterable = iterable

		def __iter__(self):
			return iter(self.iterable or [])

		def __enter__(self):
			return self

		def __exit__(self, *args):
			return False

		def update(self, *args, **kwargs):
			return None
from contextlib import contextmanager
from itertools import product
import random
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.base import (
	BaseEstimator,
	TransformerMixin
)

from sklearn.model_selection import (
	GridSearchCV,
	ParameterGrid,
)

from sklearn.metrics import (
	make_scorer,
	root_mean_squared_error,
	accuracy_score,
  	precision_score,
)

class RecipeDataPreprocessor(BaseEstimator, TransformerMixin):
	def __init__(self, blacklist = []):
		"""
		Initialize RecipeDataPreprocessor object.

		Parameters
		----------
		blacklist : list, default=[]
			List of columns to be excluded from the preprocessed data.
		"""
		self.blacklist = blacklist

	def clean_up_df(self, df: pd.DataFrame,):
		"""
		Clean up the DataFrame by dropping duplicates, stripping and converting all column names to lower case,
		and stripping leading and trailing whitespaces from the 'title' column.

		Parameters
		----------
		df : pd.DataFrame
			The DataFrame to be cleaned up.

		Returns
		-------
		pd.DataFrame
			The cleaned up DataFrame.
		"""
		df = df.drop_duplicates().copy()
		df.columns = df.columns.str.lower().str.strip()
		df['title'] = df['title'].apply(lambda x: x.strip())
		return df

	def split_df(
			self, 
			source_df: pd.DataFrame, 
			delimiters = r"(?:\s+or\s+|/|,|&)"
	):			
		df = self.clean_up_df(source_df)
		composite_cols = [col for col in df.columns if re.search(delimiters, col.lower())]

		for col in composite_cols:
			tokens = [t.strip() for t in re.split(delimiters, col.lower()) if t.strip()]
			for token in tokens:
				if token not in df.columns:
					df[token] = df[col]
			df.drop(columns=[col], inplace=True)
		return df

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		df = self.split_df(self.clean_up_df(X))
		return df.loc[:, ~df.columns.isin(self.blacklist)]

class ModelSelector():
	def __init__(self, models, scorer='rmse', pos_label=1, cv=5, n_jobs=-1, backend='threading'):
		"""
		Initialize the ModelSelector with specified models and scoring criteria.

		Parameters
		----------
		models : dict
			A dictionary where keys are model names and values are tuples 
			containing the model instance and a dictionary of hyperparameters for grid search.
		scorer : str, optional
			The scoring method to use ('rmse', 'accuracy', 'precision'), by default 'rmse'.
		pos_label : int or str, optional
			The label of the positive class when using precision as the scorer, by default 1.
		"""

		self.models = models
		self.cv = cv
		self.n_jobs = n_jobs
		self.backend = backend
		self._choose_scorer(scorer, pos_label)

	def _choose_scorer(self, scorer, pos_label):
		def precicion_single_class(y_true, y_pred, pos_label):
			y_true_bin = (y_true == pos_label)
			y_pred_bin = (y_pred == pos_label)
	
			return precision_score(y_true_bin, y_pred_bin)
	
		scorer_map = {
			'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
			'accuracy': make_scorer(accuracy_score),
			'precision': make_scorer(precicion_single_class, pos_label=pos_label),
		}
		
		self.scorer_name = scorer
		self.scorer = scorer_map.get(scorer, None)
		if self.scorer is None:
			raise ValueError(f"Unknown scorer: {scorer}")


	@contextmanager
	def _tqdm_joblib(self,tqdm_object):
		class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
			def __init__(self, *args, **kwargs):
				super().__init__(*args, **kwargs)

			def __call__(self, *args, **kwargs):
				tqdm_object.update(n=self.batch_size)
				return super().__call__(*args, **kwargs)

		old_callback = joblib.parallel.BatchCompletionCallBack
		joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
		try:
			yield
		finally:
			joblib.parallel.BatchCompletionCallBack = old_callback

	def fit(self, X_train, y_train):
		"""
		Fit the model selectors to the given data.

		Parameters
		----------
		X_train : pandas.DataFrame
			The input data.
		y_train : pandas.Series
			The target.
		"""

		self.best_models = {}
		cv_scores = {}

		for name, (model, params) in self.models.items():
			
			grid = GridSearchCV(
				model,
				params,
				scoring=self.scorer,
				cv=self.cv,
				n_jobs=self.n_jobs,
				return_train_score=False,
			)
			total_tasks = len(ParameterGrid(params)) * (grid.cv if grid.cv else self.cv)
			
			with tqdm(total=total_tasks, desc=f'Estimator: {name}', leave=True) as pbar:
				with self._tqdm_joblib(pbar):
					with joblib.parallel_backend(self.backend):
						grid.fit(X_train, y_train)

			score = grid.best_score_ * (-1 if self.scorer_name == 'rmse' else 1)
			self.best_models[name] = grid.best_estimator_
			cv_scores[name] = score

			print(f"{name}: Лучшие параметры: {grid.best_params_}\n{self.scorer_name} на кросс-валидации: {score:.4f}")

		if self.scorer_name == 'rmse':
			self.best_model_name = min(cv_scores, key=cv_scores.get)
		else:
			self.best_model_name = max(cv_scores, key=cv_scores.get)

		self.final_model = self.best_models[self.best_model_name]
		self.best_score = cv_scores[self.best_model_name]

		print(f"\nЛучшая модель: {self.best_model_name}\n{self.scorer_name} = {self.best_score:.4f}")
		return self

	def predict(self, X):
		"""
		Predict the target for the given input data.

		Parameters
		----------
		X : pandas.DataFrame
			The input data.

		Returns
		-------
		y_pred : numpy.ndarray
			The predicted target.
		"""
		if not self.final_model:
			raise BaseException('Model not choosen yet')
		with joblib.parallel_backend(self.backend):
			return self.final_model.predict(X)
	
	def final_score(self, X_test, y_test):
		"""
		Calculate and print the score of the final model on the test data.

		Parameters
		----------
		X_test : pandas.DataFrame
			The input data for test.
		y_test : pandas.Series
			The target for test.

		Returns
		-------
		None
		"""
		if not self.final_model:
			raise BaseException('Model not choosen yet')
		with joblib.parallel_backend(self.backend):
			score = self.scorer(self.final_model, X_test, y_test)
		print(f'Test {self.scorer_name} = {(score* (-1 if self.scorer_name == 'rmse' else 1)):.4f}')

class NutritionFacts():
	def __init__(
			self, 
			ingredients_list = None,
			api_key_path = None,
			drvs_path = None, 
			rdis_path = None,
			nutrients_facts_path = None,
		):
		"""
		Parameters
		----------
		ingredients_list : list
			A list of ingredients. If None, the object will try to read data from the file.
		api_key_path : str
			A path to the file with the API key.
		drvs_path : str
			A path to the file with the Daily Reference Values.
		rdis_path : str
			A path to the file with the Reference Daily Intakes.
		nutrients_facts_path : str
			A path to the file with the nutrients facts.
		
		Returns
		-------
		None
		"""
		project_root = Path(__file__).resolve().parents[1]
		datasets_dir = project_root / 'datasets'
		drvs_path = Path(drvs_path) if drvs_path else datasets_dir / 'Daily-Reference-Values-(DRVs)-under-the-New-NFL.csv'
		rdis_path = Path(rdis_path) if rdis_path else datasets_dir / 'Reference-Daily-Intakes-(RDIs)-in-the-New-Nutrition-Facts-Label.csv'
		self.nutrients_facts_path = Path(nutrients_facts_path) if nutrients_facts_path else datasets_dir / 'nutrients_facts.csv'

		self._collect_daily_nutrients(drvs_path, rdis_path)

		try:
			self.nutrients_facts = pd.read_csv(self.nutrients_facts_path, index_col=0)
		except:
			if not ingredients_list:
				raise BaseException('No ingreients to parse')
			self._headers = self._load_api_headers(api_key_path)
			self.nutrients_facts = self.calculate_percentage(self._request_to_api(ingredients_list))

	def _load_api_headers(self, api_key_path):
		api_key = os.getenv('USDA_API_KEY')
		if not api_key and api_key_path:
			with open(api_key_path, 'r') as file:
				api_key = file.read().strip()
		if not api_key:
			raise FileNotFoundError('USDA API key is required when nutrients_facts.csv is missing')
		return {
			'X-Api-Key' : api_key,
			'Content-Type' : 'application/json'
		}

	def _collect_daily_nutrients(self, drvs_path, rdis_path):
		drvs = pd.read_csv(drvs_path, sep=';')
		rdis = pd.read_csv(rdis_path, sep=';')
		nutrients = pd.concat([drvs, rdis], axis=0)
		nutrients['daily (mg)'] = self._convert_weights_to_mg(
			nutrients[[
					'Nutrient', 
					'Adults and Children ≥ 4 years', 
					'Unit of measure'
				]]
		)
		self.nutrients = nutrients[['Nutrient', 'daily (mg)']].dropna()

	def _convert_weights_to_mg(self, units_values : pd.DataFrame) -> pd.Series:
		result = units_values.copy().rename(
				columns={
					units_values.columns[0] : 'name',
					units_values.columns[1] : 'values',
					units_values.columns[2] : 'units'
				}
			)
		def _choose_unit(row):
			unit_map = {
				'g'         :   1000,
				'grams'     :   1000,
				'mg'        :   1,
				'milligrams':   1,
				'micrograms':   0.001,
				'mcg'       :   0.001,
				'ug'        :   0.001,
				'kcal'      :   1,  
			}

			iu_to_mg_map = {
				'Vitamin A'     :   0.0003,
				'Vitamin C'     :   1.0,
				'Vitamin D'     :   0.000025,
				'Vitamin E'     :   0.67,
				'Vitamin B12'   :   0.0001,  
			}
			name_match = re.match(r'^([^(,]+)', row['name'])
			name = name_match.group(1).strip()
			unit = row['units'].split()[0]
			value = float(row['values'])
			if (    row['units'].lower() == 'iu'):
				result = value * iu_to_mg_map.get(name, 0)
			else:
				result = value * unit_map.get(unit.lower(), 0)
			return result
		return result.apply(_choose_unit, axis=1)
		
	def _request_to_api(self, query_list):
		import requests

		params = {
			'api_key': self._headers['X-Api-Key'],
		}

		result = {}
		
		for query in tqdm(query_list, desc='Getting data from api'):
			json_data = {
				'pageSize' : 200,
				'query' : query,
				'dataType': [
					'Foundation',
					'SR Legacy',
					'Branded'
				],
			}
			status_code = 0
			
			while status_code != 200:
				try:
					response = requests.post(
						'https://api.nal.usda.gov/fdc/v1//foods/search',
						params=params,
						json=json_data
					)
					status_code = response.status_code
				except:
					pass
			
			json_data = response.json().get('foods')
			if len(json_data):
				json_data = json_data[0].get('foodNutrients')
			if json_data:
				data = pd.DataFrame(json_data)[['nutrientName','value','unitName']].rename(columns={'nutrientName': 'Nutrient'})
				data[query] = self._convert_weights_to_mg(data)
				result[query] = data
		return result
		
	def calculate_percentage(self, response_data):
		"""
		This function takes a response_data, which is a dictionary of pandas DataFrames,
		each of which contains the nutrient amounts for a given product. It takes the
		daily values of the nutrients from the nutrient DataFrame and calculates the
		percentage of the daily value for each nutrient. The percentages are then
		saved to a csv file.

		Parameters
		----------
		response_data : dict
			A dictionary of pandas DataFrames, each of which contains the nutrient
			amounts for a given product.

		Returns
		-------
		pandas.DataFrame
			A DataFrame with the percentage of the daily value for each nutrient.
		"""
		percents = self.nutrients.copy()
		result = {}

		def _rename_nutrient(nutrient: pd.Series):
			nutr = nutrient.copy()
			semi_ruchnya_map = {
			'Total lipid (fat)'  :  'Fat',
			'Fatty acids, total saturated'  :  'Saturated fat',
			'Carbohydrate, by difference'  :  'Total carbohydrates',
			'Fiber, total dietary'  :  'Dietary Fiber',
			'Total Sugars'  :  'Added sugars',
			'Folate, total'  :  'Folate6',
			## Not in response_data
			'Chromium'  :  'Chromium',
			'Chloride'  :  'Chloride',
			}
			nutr = nutr.map(semi_ruchnya_map).fillna(nutr)
			return nutr.apply(
				lambda x: x.replace('-','').split(',')[0].split('(')[0].strip()
			)

		for product_name, nutrient_df in response_data.items():
			nutrient_df = nutrient_df.copy()
			nutrient_df['Nutrient'] = _rename_nutrient(nutrient_df['Nutrient'])
			selected = nutrient_df[['Nutrient', product_name]]
			merged = pd.merge(percents,selected, 'left', 'Nutrient')
			merged.drop_duplicates(subset='Nutrient', keep='first', inplace=True, ignore_index=True)

			merged.set_index('Nutrient')
			
			result[product_name] = (
				merged[product_name]
				.div(merged['daily (mg)'])
				.mul(100)
			)

		frame = pd.DataFrame.from_dict(result).set_index(percents['Nutrient']).fillna(0)
		frame.to_csv(self.nutrients_facts_path) 
		return frame

	def print_facts_for_ingredients(self, ingredients_list: List[str], total = False):

		"""
		Prints the nutrition facts for the given ingredients list.
		
		Parameters:
		ingredients_list (List[str]): a list of ingredients
		total (bool): whether to print the total for all ingredients (default is False)
		
		"""
		def _print_daily_values(non_zero_nutrients):
			for k, v in non_zero_nutrients.items():
				print(f'{k} - {v:.2f}% of Daily Value')

		total_columns = [ingredient.lower() for ingredient in ingredients_list]
		facts = self.nutrients_facts[total_columns].copy()
		
		if not total:
			for ingredient in ingredients_list :
				print(ingredient.capitalize())
				filt = facts[ingredient] != 0
				non_zero_nutrients = facts[filt][ingredient]
				_print_daily_values(non_zero_nutrients.sort_index())
		else:
			facts['total'] = facts.sum(axis=1)
			_print_daily_values(facts[facts['total'] != 0]['total'].sort_index())


class DailyMenuGenerator:
    def __init__(
        self,
        df_menu,
        nutr,
        ing_cols,
        top_n_per_cat=50,
        top_k=5,
        key_nutrients=['Protein','Fat','Sodium'],
        random_samples=None
    ):
        df = df_menu[df_menu[['breakfast','lunch','dinner']].any(axis=1)].copy()
        df = df.drop_duplicates(subset='title', keep='first')
        df.set_index('title', inplace=True, drop=False)
        self.df = df

        self.nutr = nutr
        self.ing_cols = ing_cols
        self.top_n = top_n_per_cat
        self.top_k = top_k
        self.key_n = key_nutrients
        self.n_samples = random_samples

        self.cands = {}
        for cat in ['breakfast','lunch','dinner']:
            subset = self.df[self.df[cat] == 1]
            top_titles = subset.nlargest(self.top_n, 'rating').index.tolist()
            self.cands[cat] = top_titles

        self._nutr_cache = {}
        all_titles = set(self.cands['breakfast']) | set(self.cands['lunch']) | set(self.cands['dinner'])
        for title in all_titles:
            self._nutr_cache[title] = self._compute_percent(title)

    def _get_ingredients(self, title):
        row = self.df.loc[title]
        result = []
        for ing in self.ing_cols:
            if self.df.at[title, ing] == 1:
                result.append(ing)
        return result

    def _compute_percent(self, title):
        ingreds = self._get_ingredients(title)
        valid = [i for i in ingreds if i in self.nutr.nutrients_facts.columns]
        if not valid:
            return pd.Series(0, index=self.nutr.nutrients_facts.index)
        return self.nutr.nutrients_facts[valid].sum(axis=1)

    def _combo_under_limit(self, combo):
        total = pd.Series(0, index=self.nutr.nutrients_facts.index)
        for t in combo:
            total = total + self._nutr_cache[t]
            if not (total[self.key_n] <= 100).all():
                return False
        return True

    def generate(self):
        scores = []
        bs = self.cands['breakfast']
        ls = self.cands['lunch']
        ds = self.cands['dinner']

        if self.n_samples:
            for _ in range(self.n_samples):
                combo = (random.choice(bs), random.choice(ls), random.choice(ds))
                if self._combo_under_limit(combo):
                    s = sum(self.df.at[t, 'rating'] for t in combo)
                    scores.append((s, combo))
        else:
            for combo in product(bs, ls, ds):
                if self._combo_under_limit(combo):
                    s = sum(self.df.at[t, 'rating'] for t in combo)
                    scores.append((s, combo))

        if not scores:
            raise RuntimeError("Не найдено ни одной подходящей комбинации.")

        scores.sort(reverse=True, key=lambda x: x[0])
        return random.choice(scores[:self.top_k])

    def print_menu(self, combo):
        for cat, title in zip(['BREAKFAST','LUNCH','DINNER'], combo):
            row = self.df.loc[title]
            ingreds = self._get_ingredients(title)

            print(cat)
            print('-'*21)
            print(f"{row['title']} (rating: {row['rating']:.3f})")
            print("Ingredients:")
            for ing in ingreds:
                print(f"- {ing}")
            print("Nutrients:")
            cols = [i for i in ingreds if i in self.nutr.nutrients_facts.columns]
            self.nutr.print_facts_for_ingredients(cols, total=True)
            # выводим URL, если есть
            if 'epicurious_url' in row and pd.notna(row['epicurious_url']):
                print("URL:", row['epicurious_url'])
            print()

def choose_similar_recipes(
	epi_r_df : pd.DataFrame,
	similar_recipes_df: pd.DataFrame, 
	ingredients_list
) -> pd.DataFrame:
	"""
	Chooses 3 recipes from similar_recipes_df that have the most
	intersection with ingredients_list from epi_r_df.

	Parameters:
	epi_r_df (pd.DataFrame): the dataframe with the recipes
	similar_recipes_df (pd.DataFrame): the dataframe with the similar recipes
	ingredients_list (List[str]): the list of ingredients to search for

	Returns:
	pd.DataFrame: the 3 similar recipes that have the most intersection with
	ingredients_list
	"""
	merged_df = (
		epi_r_df[['title',*ingredients_list]]
		.merge(similar_recipes_df, on='title')
		.drop_duplicates()
	)
	filt = merged_df[ingredients_list].sum(axis=1) >= len(ingredients_list) / 2
	res = pd.DataFrame({})
	try:
		res = merged_df[filt].sample(3, replace=True, ).drop(ingredients_list, axis=1).fillna(0)
	except:
		pass
	return res

def print_similar_recipes(similar_recipes: pd.DataFrame):
	"""
	Prints 3 similar recipes that have the most intersection with
	ingredients_list from epi_r_df.

	Parameters:
	similar_recipes (pd.DataFrame): the dataframe with the similar recipes

	Returns:
	None
	"""
	def print_row(row : pd.DataFrame):
		print(f'\n- {row['title']}, rating: {row['epicurious_rating']}\nURL: {row['epicurious_url']}')
	similar_recipes.apply(print_row, axis=1)
