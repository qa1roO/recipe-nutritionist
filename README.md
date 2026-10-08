# Recipe Nutritionist

Recipe Nutritionist — пет-проект на Python, который оценивает идею блюда по списку ингредиентов, показывает пищевую ценность продуктов и подбирает похожие рецепты. Проект построен на данных Epicurious, классификаторе CatBoost и заранее подготовленных таблицах с нутриентами. Модель доступна через CLI и HTTP API на FastAPI, эксперименты и версии модели ведутся в MLflow, сервис собирается в Docker-образ.

## Возможности

- предсказывает класс блюда: `bad`, `so-so` или `great`;
- выводит пищевую ценность ингредиентов в процентах от дневной нормы;
- подбирает похожие рецепты с рейтингом и ссылкой на Epicurious;
- генерирует меню на день: завтрак, обед и ужин;
- отдаёт предсказание и нутриенты по HTTP (`/predict`, `/nutrition`).

## Структура проекта

```text
.
├── datasets/                 # данные рецептов, таблицы нутриентов, похожие рецепты
├── src/
│   ├── api.py                # HTTP API на FastAPI
│   ├── best_model.pkl        # обученный CatBoostClassifier
│   ├── config.py             # пути, константы, список исключаемых тегов
│   ├── nutritionist.py       # CLI-интерфейс
│   ├── recipes.py            # обработка данных, нутриенты, рекомендации
│   ├── train.py              # обучение и подбор модели с логированием в MLflow
│   └── recipes.ipynb         # исследовательский ноутбук
├── tests/                    # тесты API
├── Dockerfile
├── docker-compose.yml        # API и сервер MLflow
└── pyproject.toml            # метаданные проекта и зависимости
```

## Установка

Данные и модель хранятся в Git LFS, поэтому перед клонированием нужен `git lfs install`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[api,train,test]"
```

Группы зависимостей: `api` (FastAPI, uvicorn, клиент MLflow), `train` (MLflow), `test` (pytest), `research` (Jupyter для ноутбука).

## Использование

### CLI

```bash
python src/nutritionist.py "milk, honey, jam"
python src/nutritionist.py menu
```

В обычном режиме приложение использует кэш `datasets/nutrients_facts.csv`, поэтому ключ USDA API не нужен. Если удалить этот кэш и заново собирать данные о нутриентах, передайте ключ FoodData Central через переменную окружения `USDA_API_KEY`.

### HTTP API

```bash
uvicorn api:app --app-dir src --port 8000
```

Документация Swagger: http://localhost:8000/docs

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"ingredients": ["milk", "honey", "jam"]}'
# {"rating_class": "great", "probabilities": {"bad": 0.338, "great": 0.342, "so-so": 0.320}}
```

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | статус и источник модели |
| GET | `/ingredients` | список ингредиентов, которые знает модель |
| POST | `/predict` | класс блюда и вероятности классов |
| POST | `/nutrition` | % дневной нормы по нутриентам для каждого ингредиента |

Неизвестные ингредиенты возвращают `422` со списком того, что не распознано.

### Docker

```bash
docker compose up --build api
```

Образ содержит только то, что нужно сервису: модель, таблицу нутриентов и зависимости группы `api`.

## Обучение и MLflow

`src/train.py` воспроизводит отбор модели из ноутбука: подготовка признаков, разбиение 80/20 со стратификацией, `GridSearchCV` с 5 фолдами для Decision Tree, Random Forest и CatBoost. Каждая модель и константный бейзлайн логируются в MLflow отдельным запуском с параметрами и метриками. Лучшая модель по кросс-валидации регистрируется в Model Registry как `recipe-rating-classifier` с алиасом `champion` и сохраняется в `src/best_model.pkl`.

Локально, без сервера (запуски пишутся в `mlflow.db` и `mlruns/`):

```bash
python src/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

С сервером MLflow в Docker и API, который берёт модель из реестра:

```bash
docker compose --profile mlflow up -d mlflow              # UI: http://localhost:5000
MLFLOW_TRACKING_URI=http://localhost:5000 python src/train.py
MODEL_URI=models:/recipe-rating-classifier@champion docker compose up --build api
```

Чтобы выкатить другую версию модели, достаточно перевесить алиас `champion` в UI MLflow и перезапустить API.

## Данные и метрики

Датасет Epicurious: 18 251 рецепт, 347 бинарных признаков-ингредиентов после фильтрации тегов (кухни, праздники, способы готовки и т.п.). Рейтинг переведён в три класса: `bad` (0–1), `so-so` (2–3), `great` (4–5). Классы несбалансированы: 79% рецептов относятся к `great`.

Ошибка «плохое блюдо назвали отличным» дороже обратной, поэтому модель выбирается по **precision класса `great`**.

| Модель | precision `great`, CV | precision `great`, тест | accuracy, тест |
|---|---|---|---|
| Константа (всегда `great`) | — | 0.795 | 0.795 |
| Decision Tree | 0.840 | 0.802 | 0.659 |
| Random Forest | 0.850 | 0.849 | 0.499 |
| **CatBoost** | **0.855** | **0.851** | 0.471 |

CatBoost с `auto_class_weights='Balanced'` реже ставит `great`, зато когда ставит, ошибается заметно реже константы. Плата за это — низкая accuracy: модель чаще относит хорошие блюда к `so-so` или `bad`.

Ноутбук `src/recipes.ipynb` содержит полное исследование: регрессию рейтинга, классификацию на 6 и 3 класса, ансамбли, сбор пищевой ценности через USDA API и подготовку похожих рецептов.
