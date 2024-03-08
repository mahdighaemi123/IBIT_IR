from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
model = AutoModel.from_pretrained("HooshvareLab/bert-fa-zwnj-base")

def encode(texts, model, tokenizer):
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    return model_output.last_hidden_state[:, 0, :].numpy()

docs = {
    "عنوان خبر": "متن خبر",
    "عنوان صفحه ویکیپدیا": "متن صفحه ویکیپدیا",
    "کلمه دیکشنری": "توضیحات دیکشنری درمورد کلمه"
}

doc_keys = list(docs.keys())  
doc_embeddings = encode(doc_keys, model, tokenizer)

query = "خبر"
query_embedding = encode([query], model, tokenizer)

similarities = cosine_similarity(query_embedding, doc_embeddings)

sorted_doc_indices = similarities.argsort()[0][::-1]

print("Documents ranked by relevance to the query:")
for index in sorted_doc_indices:
    key = doc_keys[index]  
    value = docs[key]
    print(key, value)
