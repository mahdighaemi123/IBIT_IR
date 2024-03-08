from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
model = AutoModel.from_pretrained("HooshvareLab/bert-fa-zwnj-base")

df = pd.read_csv("documents.csv")
documents = {row["key"]: row["value"] for _, row in df.iterrows()}


def encode(texts, model, tokenizer):
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state[:, 0, :].numpy()


doc_keys = list(documents.keys())
doc_embeddings = encode(doc_keys, model, tokenizer)


@app.route('/retrieve', methods=['POST'])
def retrieve_documents_func():
    data = request.json
    query = data.get('query')
    k = data.get('k', 3)

    if not query:
        return jsonify({"error": "Query is required."}), 400

    query_embedding = encode([query], model, tokenizer)
    similarities = cosine_similarity(query_embedding, doc_embeddings)

    sorted_doc_indices = similarities.argsort()[0][::-1]
    sorted_documents = [documents[doc_keys[index]]
                        for index in sorted_doc_indices]

    retrieve_documents = sorted_documents[:k]

    return jsonify(retrieve_documents)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)
