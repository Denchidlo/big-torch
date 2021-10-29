from big_torch_neptune import NeptuneClient
import json

with open('./config/credentials.json', 'r+') as reader:
    creds = json.load(reader)

client = NeptuneClient(creds)

client.process_experiment('./config/model.json')