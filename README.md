# Homecraft Retail demo with Elastic ESRE and Google Cloud's Gemini

This repo shows how to leverage Elastic search capabilities (both text and vector ones) togheter with Google Cloud's GenerativeAI model (Gemini-pro) and VertexAI features to create a new retail experience. With this repo you will:

- Create a python streamlit app with an intelligent search bar
- Integrate with Gemini models and VertexAI APIs
- Configure an Elastic cluster as a private data source to build context for LLMs
- Ingest data from multiple data sources (Web Crawler, files, BigQuery)
- Use Elastic's text_embeddings and vector search for finding relevant content
- and more...

too see details around how to configure all the components have a look at this repo [here](https://github.com/valerioarvizzigno/homecraft_vertex_lab). It refers to the Palm2 version of this demo, but it's still valid because only a few lines regarding LLM model call/init change.

## Sample questions

---USE THE HOME PAGE FOR BASE DEMO---

Try queries like: 

- "List the 3 top paint primers in the product catalog, specify also the sales price for each product and product key features. Then explain in bullet points how to use a paint primer".
You can also try asking for related urls and availability --> leveraging private product catalog + public knowledge

- "could you please list the available IKEA stores in UK" --> --> it will likely use (crawled docs)

- "Which are the ways to contact IEKA customer support in the UK? What is the webpage url for customer support?" --> it will likely use crawled docs

- Please provide the social media accounts info from the company --> it will likely use crawled docs

- Please provide the full address of the Manchester store in UK --> it will likely use crawled docs

- are you offering a free parcel delivery? --> it will likely use crawled docs

- Could you please list my past orders? Please specify price for each product --> it will search into BigQuery order dataset

- Which product are available in the product catalog for the Bathroom category? Give a short description of each product

- List all the items I have bought in my order history in bullet points

