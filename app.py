import streamlit as st
from dotenv import load_dotenv
import json
import os, time
import uuid

from retrieval_pipeline import get_retriever, get_compression_retriever
from retrieval_pipeline.cache import SemanticCache
import benchmark


def get_result(query, retriever, use_cache):
    t0 = time.time()
    retrieved_chunks = retriever.get_relevant_documents(query, use_cache=use_cache)
    latency = time.time() - t0
    return retrieved_chunks, latency
 
st.set_page_config(
    layout="wide",
    page_title="Retrieval Demo"
)


@st.cache_resource
def setup_retriever():
    load_dotenv()
    ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')

    retriever = get_retriever(index='masa.ai', elasticsearch_url=ELASTICSEARCH_URL)
    compression_retriever = get_compression_retriever(retriever)
    semantic_cache_retriever = SemanticCache(compression_retriever)
    return semantic_cache_retriever 


def retrieval_page(retriever, use_cache):
    with st.form(key='input_form'):
        query_input = st.text_area("Query Input")
        submit_button = st.form_submit_button(label='Retrieve')
    
    if submit_button:
        with st.spinner('Processing...'):
            result, latency = get_result(query_input, retriever=retriever, use_cache=use_cache)
            st.subheader("Please find the retrieved documents below 👇")
            st.write("latency:", latency, " s")
            st.json(result)



def main():
    st.title("Part 3: Search")
    use_cache = st.sidebar.toggle("Use cache", value=True)

    st.sidebar.info("""
**Retrieval Pipeline Evaluation Result:**
- **MRR**: 0.756
- **Avg. Latency**: 4.50s (on CPU, with cache turned off)
- **Benchmark Result**: https://docs.google.com/spreadsheets/d/1WJnb8BieoxLch0gvb53ZzMS70r_G35PKm731ubdeNCA/edit?usp=sharing
""")

    with st.spinner('Setting up...'):
        retriever = setup_retriever()


    retrieval_page(retriever, use_cache)

    # with st.expander("Tech Stack Used"):
    #     st.markdown("""
    #     **Flash Rank**: Ultra-lite & Super-fast Python library for search & retrieval re-ranking.

    #     - **Ultra-lite**: No heavy dependencies. Runs on CPU with a tiny ~4MB reranking model.
    #     - **Super-fast**: Speed depends on the number of tokens in passages and query, plus model depth.
    #     - **Cost-efficient**: Ideal for serverless deployments with low memory and time requirements.
    #     - **Based on State-of-the-Art Cross-encoders**: Includes models like ms-marco-TinyBERT-L-2-v2 (default), ms-marco-MiniLM-L-12-v2, rank-T5-flan, and ms-marco-MultiBERT-L-12.
    #     - **Sleek Models for Efficiency**: Designed for minimal overhead in user-facing scenarios.

    #     _Flash Rank is tailored for scenarios requiring efficient and effective reranking, balancing performance with resource usage._
    #     """)

if __name__ == "__main__":
    main()