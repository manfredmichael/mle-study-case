from langchain.vectorstores import ElasticVectorSearch
from langchain.llms import OpenAI, HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from retrieval_pipeline.hybrid_search import HybridRetriever, get_dense_db, get_sparse_retriever
from retrieval_pipeline.utils import get_hybrid_indexes

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import logging
import tqdm


def get_compression_retriever(retriever):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

# Embedding Models Loader
def get_huggingface_embeddings(model_name):
    logging.info(f"Loading Huggingface Embedding")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def get_vectorstore(index_name, embeddings, elasticsearch_url=None):
    logging.info(f"Loading vectorstore")

    index_dense, index_sparse = get_hybrid_indexes(index_name)

    dense_db = get_dense_db(elasticsearch_url, index_dense, embeddings)
    dense_retriever = dense_db.as_retriever()

    sparse_retriever = get_sparse_retriever(elasticsearch_url, index_sparse)

    hybrid_retriever = HybridRetriever(
        dense_db=dense_db,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        index_dense=index_dense,
        index_sparse=index_sparse,
        top_k_dense=2,
        top_k_sparse=3
    )

    # db = ElasticVectorSearch(
    #     elasticsearch_url=elasticsearch_url,
    #     index_name=index_name,
    #     embedding=embeddings,
    # )
    return hybrid_retriever

def get_retriever(index, elasticsearch_url):
    # cache.init(pre_embedding_func=get_msg_func)
    # cache.set_openai_key(openai_api_key)

    embeddings = get_huggingface_embeddings(model_name="multilingual-e5-large")

    # llm = get_openai_llm(
    #     model_name=model_name, temperature=0, api_key=model_api_key
    # )
    # embeddings = get_openai_embeddings(embedding_api_key, embedding_name)

    # question_generator = load_question_generator(llm)
    # answer_generator = load_answer_generator(llm, company=model_config['company_name'], tone=model_config['tone'], additional_instructions=model_config['additional_instructions'])

    retriever = get_vectorstore(
        index,
        embeddings=embeddings,
        elasticsearch_url=elasticsearch_url,
    )

    # if history:
    #     qa = get_conversational_chain(retriever, question_generator, answer_generator)
    # else:
    #     qa = get_retrieval_chain(retriever, answer_generator)

    # chain = CustomLLMChain(
    #     chain=qa,
    #     model_name=llm.model_name,
    #     use_history=history
    # )
    # 
    # 
    return retriever

def get_relevant_documents(query, retriever, top_k):
    results = retriever.get_relevant_documents(query)
    passages = [{
        "id": i,
        "text": result.page_content
    } for i, result in enumerate(results)]

    reranked_result = ranker.rerank(RerankRequest(query=query, passages=passages))
    return reranked_result 
