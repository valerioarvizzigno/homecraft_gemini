import os
import streamlit as st
from elasticsearch import Elasticsearch
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, GenerationConfig

# This code shows VertexAI GenAI integration with Elastic Vector Search features
# to connect publicly trained LLMs with private data
# Gemini Pro model is used

# Code is presented for demo purposes but should not be used in production
# You may encounter exceptions which are not handled in the code


# Required Environment Variables
# gcp_project_id - Google Cloud project ID
# cloud_id - Elastic Cloud Deployment ID
# cloud_user - Elasticsearch Cluster User
# cloud_pass - Elasticsearch User Password

projid = os.environ['gcp_project_id']
cid = os.environ['cloud_id']
cp = os.environ['cloud_pass']
cu = os.environ['cloud_user']

generation_config = GenerationConfig(
    temperature=0.4, # 0 - 1. The higher the temp the more creative and less on point answers become
    max_output_tokens=2048, #modify this number (1 - 1024) for short/longer answers
    top_p=0.8,
    top_k=40,
    candidate_count=1,
)

#            safety_settings={
#                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#            }

vertexai.init(project=projid, location="us-central1")

model = GenerativeModel("gemini-pro")
#Gemini can hold chat history in the chat variable. Pass this variable every time along with the prompt one. 
#chat = model.start_chat()

# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    return es

# Search ElasticSearch index and return details on relevant products
def search_products(query_text):

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "title-vector"
                }
            }]
        }
    }

    knn = {
        "field": "title-vector",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "description", "url", "availability", "price", "brand", "product_id"]
    index = 'home-depot-product-catalog-vector'
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=5,
                     source=False)

    doc_list = resp['hits']['hits']
    body = resp['hits']['hits']
    url = ''
    for doc in doc_list:
        #body = body + doc['fields']['description'][0]
        url = url + "\n\n" +  doc['fields']['url'][0]

    return body, url

# Search ElasticSearch index and return body and URL for crawled docs
def search_docs(query_text):
    

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "title-vector"
                }
            }]
        }
    }

    knn = {
        "field": "title-vector",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "body_content", "url"]
    index = 'search-homecraft-ikea'
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=1,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    return body, url

# Search ElasticSearch index for user's order history
def search_orders(user):

    # Use only text-search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "user_id": {
                        "query": user,
                        "boost": 1
                    }
                }
            }]
        }
    }

    fields = ["id", "order_id", "user_id", "product_id" "status", "created_at", "shipped_at", "delivered_at", "returned_at", "sale_price"]
    index = 'bigquery-thelook-order-items'
    resp = es.search(index=index,
                     query=query,
                     fields=fields,
                     size=10,
                     source=False)

    order_items_list = resp['hits']['hits']

    return order_items_list

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from Gemini based on the given prompt
def vertexAI(chat: ChatSession, prompt: str) -> str:
    response = chat.send_message(prompt)
    return response.text

def generateResponse(prompt):
    response = model.generate_content(prompt, 
                                      generation_config=generation_config
    )
    return response.text


#image = Image.open('homecraft_logo.jpg')
st.image("https://i.imgur.com/cdjafe0.png", caption=None)
st.title("HomeCraft Search Bar")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("You: ")
    submit_button = st.form_submit_button("Send")

# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from Homecraft dataset."
if submit_button:
    es = es_connect(cid, cu, cp)
    resp_products, url_products = search_products(query)
    resp_docs, url_docs = search_docs(query)
    resp_order_items = search_orders(1) # 1 is the hardcoded userid, to simplify this scenario. You should take user_id by user session
    #prompt = f"You are an e-commerce AI assistant. Answer this question: {query} using this context: \n{resp_products} \n {resp_docs} \n {resp_order_items}."
    #prompt = f"You are an e-commerce AI assistant. Answer this question: {query}.\n If product information is requested use the information provided in this JSON: {resp_products} listing the identified products in bullet points with this format: Product name, product key features, price, web url. \n For other questions use the documentation provided in these docs: {resp_docs} and your own knowledge. \n If the question contains requests for user past orders consider the following order list: {resp_order_items}"
    prompt = [f"You are an e-commerce AI assistant.", f"You answer question around product catalog, general company information and user past orders", f"Answer this question: {query}.", f"Context: Product catalog = {resp_products}; General information = {resp_docs}; Past orders = {resp_order_items} "]
    #prompt = f"You are an e-commerce customer AI assistant. Answer this question: {query}.\n with your own knowledge and using the information provided in the context. Context: JSON product catalog: {resp_products} \n, these docs: \n {resp_docs} \n and this user order history: \n {resp_order_items}"
    #answer = vertexAI(chat, prompt)
    answer = generateResponse(prompt)

    if answer.strip() == '':
        st.write(f"Search Assistant: \n\n{answer.strip()}")
    else:
        st.write(f"Search Assistant: \n\n{answer.strip()}\n\n")
        st.write(f"Order items: {resp_order_items}")

