from dotenv import load_dotenv
import json
import os, time
import uuid

from retrieval_pipeline import get_retriever, get_compression_retriever
import benchmark
from retrieval_pipeline.hybrid_search import store

from retrieval_pipeline.cache import SemanticCache

load_dotenv()
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')
# HUGGINGFACE_KEY = os.getenv('HUGGINGFACE_KEY')

os.environ["ES_ENDPOINT"] = ELASTICSEARCH_URL
print(ELASTICSEARCH_URL)

if __name__ == "__main__":
    retriever = get_retriever(index='masa.ai', elasticsearch_url=ELASTICSEARCH_URL)
    compression_retriever = get_compression_retriever(retriever)

    semantic_cache_retriever = SemanticCache(compression_retriever)

    t0 = time.time() 
    retrieved_chunks = compression_retriever.get_relevant_documents('Gunung Semeru')
    t = time.time() - t0
    print(retrieved_chunks)

    # benchmark.get_benchmark_result("benchmark-reranker.csv", retriever=compression_retriever)

    # for i in range(100):
    #     query = input("query: ")
    #     t0 = time.time()
    #     # retrieved_chunks = compression_retriever.get_relevant_documents(query)
    #     retrieved_chunks = semantic_cache_retriever.get_relevant_documents(query)

    #     t = time.time() - t0

    #     print(list(store.yield_keys()))
    #     print('time:', t)

    #     print("Result:")
    #     for r in retrieved_chunks:
    #         print(r.page_content[:50])
    #     print()