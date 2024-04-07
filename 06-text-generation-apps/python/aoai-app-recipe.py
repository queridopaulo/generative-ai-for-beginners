import os
from ollama import Client
from dotenv import load_dotenv
load_dotenv()

client = Client(
    host = os.environ['OLLAMA_ENDPOINT']
    )
model = "mistral" # Change the model accordingly

no_recipes = input("No of recipes (for example, 5: ")

ingredients = input("List of ingredients (for example, chicken, potatoes, and carrots: ")

filter = input("Filter (for example, vegetarian, vegan, or gluten-free: ")

# interpolate the number of recipes into the prompt an ingredients
prompt = f"Show me {no_recipes} recipes for a dish with the following ingredients: {ingredients}. Per recipe, list all the ingredients used, no {filter}: "
messages = [{"role": "user", "content": prompt}]
options = {
  "temperature": 0.8,
  "num_predict": 1200
}

completion = client.chat(model=model, messages=messages, options=options)


# print response
print("Recipes:")
print(completion['message']['content'])

old_prompt_result = completion['message']['content']
prompt_shopping = "Produce a shopping list, and please don't include ingredients that I already have at home: "

new_prompt = f"Given ingredients at home {ingredients} and these generated recipes: {old_prompt_result}, {prompt_shopping}"
messages = [{"role": "user", "content": new_prompt}]
options = {
  "temperature": 0,
  "num_predict": 50
}
completion = client.chat(model=model, messages=messages, options=options)

# print response
print("\n=====Shopping list ======= \n")
print(completion['message']['content'])

