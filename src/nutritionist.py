#!/usr/bin/env python

from recipes import *

import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / 'datasets'
SRC_DIR = PROJECT_ROOT / 'src'

def main():
	if len(sys.argv) < 2:
		raise SystemExit('Usage: python src/nutritionist.py "milk, honey, jam" | menu')

	try:
		source_df = pd.read_csv(DATASETS_DIR / 'epi_r.csv', encoding='utf-8')
	except:
		file = 'https://drive.google.com/uc?export=download&id=1hzmxNBrY7-9mv5EpqAvhVUiJahfrcYUN'
		source_df = pd.read_csv(file, encoding="utf-8")
		source_df['title'] = source_df['title'].apply(lambda x: x.strip())
		source_df.to_csv(DATASETS_DIR / 'epi_r.csv', index=False)

	similar_recipes_df = pd.read_csv(DATASETS_DIR / 'similar_recipes.csv').rename(columns={'recipe': 'title'})
	model = joblib.load(SRC_DIR / 'best_model.pkl')
	all_ingredients = list(getattr(model, 'feature_names_in_', []))
	if not all_ingredients:
		all_ingredients = pd.read_csv(DATASETS_DIR / 'nutrients_facts.csv', nrows=0).columns.drop('Nutrient').tolist()
	filtered_df = prepare_feature_df(source_df, all_ingredients)

	if sys.argv[1].lower() == 'menu':
		bonus_part( source_df,filtered_df, similar_recipes_df)
	else:
		ingredients_list = parse_ingredients(sys.argv[1:])
		check_ingredients(ingredients_list, all_ingredients)
		base_nutritionist(sorted(ingredients_list), all_ingredients, filtered_df, similar_recipes_df, model)

	return 0

def parse_ingredients(args):
	ingredients = []
	for arg in args:
		ingredients.extend(arg.split(','))
	return [ingredient.strip().lower() for ingredient in ingredients if ingredient.strip()]

def prepare_feature_df(source_df, all_ingredients):
	preprocessed = RecipeDataPreprocessor().split_df(source_df)
	for ingredient in all_ingredients:
		if ingredient not in preprocessed.columns:
			preprocessed[ingredient] = 0
	metadata = [col for col in ['title', 'rating', 'breakfast', 'lunch', 'dinner'] if col in preprocessed.columns]
	return preprocessed[metadata + list(all_ingredients)].copy()

def check_ingredients(ingredients_list, all_ingredients):
	for i in ingredients_list:
		if i not in all_ingredients:
			raise AttributeError(f'Ingredient {i} not in dataset')

def base_nutritionist(ingredients_list, all_ingredients, source_df, similar_recipes_df, model):

	predict_vector = pd.DataFrame({f: [0] for f in all_ingredients})
	predict_vector[ingredients_list] = 1
	
	predict = np.ravel(model.predict(predict_vector))
	nutr = NutritionFacts(ingredients_list)
	similar_recipes = choose_similar_recipes(source_df, similar_recipes_df, ingredients_list).drop_duplicates()

	print('\nI. OUR FORECAST')
	print(f'You might find it tasty, but in our opinion, it is a {predict[0]} idea to have a dish with that list of ingredients.')
	print('\nII. NUTRITION FACTS')
	nutr.print_facts_for_ingredients(ingredients_list)
	print(f'\nIII. {len(similar_recipes)} SIMILAR RECIPES:')
	print_similar_recipes(similar_recipes)

def prepare_menu_df(source_df, filtered_df, similar_recipes_df):
	df_menu = filtered_df.copy()
	df_menu['title'] = source_df['title']
	df_menu['dinner'] = source_df['dinner']
	df_menu['lunch'] = source_df['lunch']
	df_menu['breakfast'] = source_df['breakfast']
	df_menu = df_menu[(df_menu['breakfast'] == 1) | (df_menu['lunch'] == 1) | (df_menu['dinner'] == 1)]
	df_menu = df_menu.merge(similar_recipes_df[['title', 'epicurious_url']], how='inner', on='title').dropna()

	return df_menu

def bonus_part( source_df,filtered_df, similar_recipes_df):
	df_menu = prepare_menu_df(source_df, filtered_df, similar_recipes_df)
	nutr = NutritionFacts(filtered_df.drop('rating', axis=1).columns.tolist())
	reserved = {'title','rating','breakfast','lunch','dinner','epicurious_url'}
	ing_cols = [
		c for c in df_menu.columns
		if c not in reserved
		and pd.api.types.is_numeric_dtype(df_menu[c])
		and set(df_menu[c].dropna().unique()).issubset({0,1})
	]

	gen = DailyMenuGenerator(
		df_menu,
		nutr,
		ing_cols,             
		top_n_per_cat=50,
		top_k=5,
		key_nutrients=['Protein','Fat','Sodium'],
		random_samples=10000
	)
	score, combo = gen.generate()
	gen.print_menu(combo)

if __name__ == '__main__':
	main()
