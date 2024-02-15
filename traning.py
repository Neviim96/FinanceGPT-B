import json
import openai
import os
import pandas as pd
from pprint import pprint
from openai import OpenAI

client = OpenAI()
# Use API

client = OpenAI(api_key="please write your OpenAI key") #please write your OpenAI key eg:sk-xxxxxxxxxxx

# Upload traning data
file_response = client.files.create(
    file=open(r"please write your traning data address", "rb"), #please write your traning data address eg:C:\Users\...\xxx.jsonl. 
    purpose="fine-tune"
)

training_file_name_K=file_response.id
print(training_file_name_K)

#Traning
traning_model_K=client.fine_tuning.jobs.create(
  training_file=training_file_name_K, 
  model="gpt-3.5-turbo-1106", 
  hyperparameters={
    "n_epochs":10
  }
)

print(traning_model_K.id)
print(traning_model_K.fine_tuned_model)