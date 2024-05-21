import faiss
from sentence_transformers import SentenceTransformer
import time
import json

from langchain_core.documents import Document

def init_cache():
    index = faiss.IndexFlatL2(1024)
    if index.is_trained:
        print("Index trained")

    # Initialize Sentence Transformer model
    encoder = SentenceTransformer("multilingual-e5-large")

    return index, encoder

def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {"query": [], "embeddings": [], "answers": [], "response_text": []}

    return cache

def store_cache(json_file, cache):
    with open(json_file, "w") as file:
        json.dump(cache, file)

class SemanticCache:
    def __init__(self, retriever, json_file="cache_file.json", thresold=0.35):
        # Initialize Faiss index with Euclidean distance
        self.retriever = retriever
        self.index, self.encoder = init_cache()

        # Set Euclidean distance threshold
        # a distance of 0 means identicals sentences
        # We only return from cache sentences under this thresold
        self.euclidean_threshold = thresold

        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)

    def query_database(self, query_text):
        results = self.retriever.get_relevant_documents(query_text)
        return results

    def get_relevant_documents(self, query: str, use_cache=True) -> str:
        # Method to retrieve an answer from the cache or generate a new one
        start_time = time.time()
        try:
            # First we obtain the embeddings corresponding to the user query
            embedding = self.encoder.encode([query])

            # Search for the nearest neighbor in the index
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)

            if use_cache:
                if D[0] >= 0:
                    if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                        row_id = int(I[0][0])

                        print("Answer recovered from Cache. ")
                        print(f"{D[0][0]:.3f} smaller than {self.euclidean_threshold}")
                        print(f"Found cache in row: {row_id} with score {D[0][0]:.3f}")

                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Time taken: {elapsed_time:.3f} seconds")
                        return [Document(**doc) for doc in self.cache["answers"][row_id]]

            # Handle the case when there are not enough results
            # or Euclidean distance is not met, asking to chromaDB.
            answer = self.query_database(query)
            # response_text = answer["documents"][0][0]

            self.cache["query"].append(query)
            self.cache["embeddings"].append(embedding[0].tolist())
            self.cache["answers"].append([doc.__dict__ for doc in answer])


            self.index.add(embedding)
            store_cache(self.json_file, self.cache)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")

            return answer 
        except Exception as e:
            raise RuntimeError(f"Error during 'get_relevant_documents' method: {e}")