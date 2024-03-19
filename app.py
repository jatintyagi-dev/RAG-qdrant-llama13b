
import streamlit as st
import pandas as pd
import numpy as np

from torch import cuda, bfloat16
import transformers


model_id = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = ''
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.LlamaForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth,
) 

tokenizer = transformers.AutoTokenizer.from_pretrained(
                        model_id,
                        use_auth_token=hf_auth)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

import torch

query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto")

from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=query_pipeline)


        
import qdrant_client

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

client = qdrant_client.QdrantClient(
    "<qdrant-url>",
    api_key="<qdrant-api-key>", # For Qdrant Cloud, None for local instance
)

doc_store = Qdrant(
    client=client, collection_name="texts", 
    embeddings=embeddings,
)
from langchain.chains import VectorDBQA, RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=doc_store.as_retriever(),
)



def main():
    

    st.title("Large-Scale RAG Application")

    tab1, tab2 = st.tabs([ "Ask Question","Browse Context"])
    context = st.empty()

    

    with tab1:
       
          query = st.text_input("Ask questions")
  
        if query:
            
            context = doc_store.similarity_search_with_score(query)
             
            response = qa.run(input_documents = docs, question = query)
                 
            st.write(response)
            
    with tab2:
        st.write("Context browsing interface here.")
        st.markdown(" <br /> ".join([c[0].page_content for c in context])
        #upload a your pdf file
        

if __name__=="__main__":
    main()
