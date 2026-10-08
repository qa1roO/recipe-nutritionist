from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / 'datasets'
SRC_DIR = PROJECT_ROOT / 'src'

MODEL_PATH = SRC_DIR / 'best_model.pkl'
NUTRIENTS_FACTS_PATH = DATASETS_DIR / 'nutrients_facts.csv'
RECIPES_PATH = DATASETS_DIR / 'epi_r.csv'

REGISTERED_MODEL_NAME = 'recipe-rating-classifier'
TARGET_CLASS = 'great'
RANDOM_STATE = 21

# Non-ingredient tags of the Epicurious dataset: nutrients, meal types, cooking methods,
# holidays, places, authors and dish types. The model sees ingredients only.
BLACKLIST = [
	"calories", "protein", "fat", "sodium", "#cakeweek", "#wasteless",
	"22-minute meals", "3-ingredient recipes", "30 days of groceries", "advance prep required", "breakfast", "brunch", "dinner",
	"lunch", "dessert", "appetizer", "snack", "snack week", "side", "quick","easy", "quick and healthy",
	"advance prep required", "kid-friendly", "vegan", "vegetarian", "pescatarian", "low cal", "low fat",
	"low sugar", "low","no sugar", "low carb", "low cholesterol", "low sodium", "fat free", "pot pie",
	"sugar conscious", "healthy", "high fiber", "dairy free", "peanut free", "tree nut free", "soy free",
	"kosher", "kosher for passover", "wheat","gluten-free", "no sugar added", "non-alcoholic", "alcoholic", "organic",
	"tested","improved", "no-cook", "freeze","chill", "freezer food", "raw", "grill", "grill","barbecue",
	"broil", "roast", "bake", "pan-fry", "deep-fry", "fry", "simmer","fruit","braise",
	"steam", "stir-fry", "slow cooker", "pressure cooker", "smoker", "microwave", "food processor",
	"blender", "potato salad", "double boiler", "ice cream machine", "juicer", "mandoline", "mortar and pestle",
	"candy thermometer", "cookbook critic", "paleo", "cook like a diner", "house cocktail", "epi loves the microwave", "epi + ushg",
	"sandwich theory", "friendsgiving", "family reunion", "anniversary", "birthday", "wedding", "engagement party",
	"party", "picnic", "buffet", "potluck", "tailgating", "cocktail party", "super bowl",
	"thanksgiving", "christmas", "christmas eve", "new year's eve", "new year's day", "meatball", "meatloaf",
	"shower", "ramekin", "valentine's day", "st. patrick's day", "halloween", "hanukkah", "rosh hashanah","yom kippur",
	"purim", "sukkot", "passover", "shavuot", "diwali", "kwanzaa", "ramadan",
	"easter", "fourth of july", "bastille day", "mardi gras", "lunar new year", "oscars", "father's day",
	"winter", "flaming hot summer", "back to school", "backyard bbq", "entertaining", "casserole","gratin", "pasta maker",
	"pressure cooker", "sandwich","sourdough", "d.c.", "lasagna", "hot drink",
	"house","garden", "quiche", "frittata", "fritter", "salad",
	"salad dressing", "pizza", "cupcake", "cake", "cookie", "cookies", "brownie",
	"muffin", "biscuit", "waffle", "pancake", "taco", "burrito", "omelet","stew","soup","stew",
	"smoothie", "ice cream", "sorbet", "crêpe", "cobbler","crumble", "custard",
	"candy", "marshmallow", "edible gift", "hors d'oeuvre", "condiment", "condiment","spread", "salsa",
	"sauce", "rub", "marinade", "marinate", "pickles", "parade",
	"suzanne goin", "dorie greenspan", "dip", "drink", "drinks", "aperitif", "digestif",
	"cocktail", "spirit", "connecticut", "dallas", "denver", "stuffing","dressing", "spring",
	"summer", "fall", "graduation", "pasadena", "boston", "beverly hills", "nancy silverton",
	"oktoberfest", "poker", "game night", "pie", "providence", "france", "italy", "spain",
	"germany", "switzerland", "australia", "england", "santa monica", "kansas city", "kentucky derby",
	"kidney friendly", "kitchen olympics", "idaho", "hummus", "minneapolis", "mixer", "no meat","no problem",
	"persian new year", "sauté", "self", "ireland", "israel", "japan", "egypt",
	"canada", "haiti", "jamaica", "mexico", "peru", "philippines", "bulgaria",
	"cuba", "dominican republic", "georgia", "guam", "arizona", "alabama", "alaska",
	"california", "colorado", "mother's day", "one-pot meal", "punch", "skewer", "florida",
	"hawaii", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
	"maine", "maryland", "massachusetts", "labor day", "michigan", "minnesota", "mississippi",
	"missouri", "nebraska", "new hampshire", "new jersey", "new mexico", "new orleans", "cambridge",
	"new york", "bon appétit", "harpercollins", "gourmet", "frankenrecipe", "frozen dessert", "game",
	"bon app��tit", "north carolina", "ohio", "weelicious", "chill", "oklahoma", "oregon",
	"pennsylvania", "rhode island", "south carolina", "tennessee", "texas", "utah", "vermont",
	"virginia", "washington", "west virginia", "wisconsin", "las vegas", "los angeles", "brooklyn",
	"atlanta", "columbus", "anthony bourdain", "houston", "emeril lagasse", "miami", "long beach",
	"san francisco", "seattle", "portland", "pittsburgh", "louisville", "windsor", "westwood",
	"lancaster", "healdsburg", "yonkers", "paris", "aspen", "costa mesa", "pacific palisades",
	"st. louis", "cookbooks", "leftovers", 'title', 'camping', 'chicago', 'hollywood', 'jalapeño',
	'seafood', 'sherry', 'soufflé', 'spritzer', 'triple sec', 'vermouth', 'yuca',
	'campari', 'chartreuse', 'grappa', 'jícama', 'kahlúa', 'marscarpone', 'rosé',
]
