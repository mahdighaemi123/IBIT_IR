# IBIT_IR
persian information retrieval based on `HooshvareLab/bert-fa-zwnj-base`

## data
save your data as key,value in documents.csv

## run
python server.py

## connect with curl
```
curl --location 'http://127.0.0.1:5010/retrieve' \
--form 'query="خبر"'
```

## connect with py
```
import requests

url = 'http://127.0.0.1:5010/retrieve'

data = {
    'query': 'news',
    'k': 2  # Optional
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Request successful.")
    print("Response:")
    print(response.json())  
else:
    print(f"Request failed with status code {response.status_code}.")
    print("Response:")
    print(response.text)  
```