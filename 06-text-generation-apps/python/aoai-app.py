import os
from ollama import Client
from dotenv import load_dotenv
load_dotenv()

model = "mistral" # Change the model accordingly

# configure OLLaMa client 
client = Client(host = os.environ['OLLAMA_ENDPOINT'])

# add your completion code
prompt = "Complete the following: Once upon a time there was a"
messages = [{"role": "user", "content": prompt}]  
# make completion
completion = client.chat(model=model, messages=messages)

# print response
print(completion['message']['content'])

#  very unhappy _____.

# Once upon a time there was a very unhappy mermaid.
