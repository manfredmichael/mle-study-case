from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_community.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import elasticsearch


from typing import Optional, List

from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

store = LocalFileStore("cache")

class HybridRetriever(BaseRetriever):
    dense_db: ElasticVectorSearch
    dense_retriever: VectorStoreRetriever
    sparse_retriever: ElasticSearchBM25Retriever
    index_dense: str
    index_sparse: str
    top_k_dense: int
    top_k_sparse: int

    is_training: bool = False
    
    @classmethod
    def create(
        cls, dense_db, dense_retriever, sparse_retriever, index_dense, index_sparse, top_k_dense, top_k_sparse
        ):

        return cls(
                dense_db=dense_db,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                index_dense=index_dense,
                index_sparse=index_sparse,
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse,
            )

    def reset_indices(self):
        result = self.dense_db.client.indices.delete(
            index=self.index_dense,
            ignore_unavailable=True,
            allow_no_indices=True,
        )

        
        logging.info('dense_db delete:', result)

        result = self.sparse_retriever.client.indices.delete(
            index=self.index_sparse,
            ignore_unavailable=True,
            allow_no_indices=True,
        )

        logging.info('sparse_retriever delete:', result)

    def add_documents(self, documents, batch_size=25):
        for i in range(0, len(documents), batch_size):
            print('batch', i) 
            dense_batch = documents[i:i + batch_size]
            sparse_batch = [doc.page_content for doc in dense_batch]
            self.dense_retriever.add_documents(dense_batch)
            self.sparse_retriever.add_texts(sparse_batch)
        
    def _get_relevant_documents(self, query: str, **kwargs):
        dense_results = self.dense_retriever.get_relevant_documents(query)[:self.top_k_dense]
        sparse_results = self.sparse_retriever.get_relevant_documents(query)[:self.top_k_sparse]

        # Combine results (you'll need a strategy here) 
        combined_results = dense_results + sparse_results

        # Create LangChain Documents
        documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in combined_results]
        # documents = [Document(page_content=doc.page_content, metadata=doc.metadata, relevance_score=result.relevance_score) for result, doc in zip(reranked_result, combined_results)]
        return documents
    
    async def aget_relevant_documents(self, query: str): 
        raise NotImplementedError 

def get_dense_db(elasticsearch_url, index_dense, embeddings):
    # retriever cache
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, 
        namespace='sentence-transformer',
        # query_embedding_store=store,
        # query_embedding_cache=True
    )

    cached_embedder.query_embedding_store = store

    dense_db = ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name=index_dense,
        embedding=embeddings,
        # embedding=cached_embedder,
    )
    return dense_db
    
def get_sparse_retriever(elasticsearch_url, index_sparse):
    sparse_retriever = ElasticSearchBM25Retriever(client=elasticsearch.Elasticsearch(elasticsearch_url), 
                                                  index_name=index_sparse)
    return sparse_retriever
