# RAG-Qdrant-LLAMA
> Create a retrieval augmented generation using Qdrant as vector DB and Llama-2-13B-chat as LLM 

Refer this article: [Building Large scale RAG application with Qdrant and LLAMA  ](https://medium.com/@im_jatintyagi/building-large-scale-rag-applications-using-llama-2-13b-and-qdrant-e583f235154d)


## For Running Application in kaggle or colab
* Deploy the docker container for running the Qdrant server
  ```python
  !docker run -p "6333:6333" -p "6334:6334" --name "rag-llm-qdrant" --rm -d qdrant/qdrant:latest
  ```


* Clone this Repo
  
* replace the hf_auth in line 25 of app.py
  
* replace the qdrant url in line 70 of app.py to "http://localhost:6333" & api_key to `None`

* Run Streamlit server 
 
```python
!pip install streamlit
!streamlit run --server.address 0.0.0.0 --server.port 80 app.py 
```




## For accessing the app using ngrok

```python
 !pip install ngrok
# import ngrok python sdk
import ngrok

# Establish connectivity
listener = ngrok.forward("http://0.0.0.0:80", authtoken ="" #replace with auth token from ngrok )

```

