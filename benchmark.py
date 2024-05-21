import pandas as pd 
from retrieval_pipeline import get_relevant_documents 
import tqdm, time


TOP_N = 3

def get_benchmark_result(path, retriever):
    df = pd.read_csv(path)

    retrieval_result = []
    query_result = [[] for i in range(TOP_N)] 
    retrieval_latency = []

    # j = 0
    for i, row in tqdm.tqdm(df.iterrows()):
        # j+=1
        query = row['query']
        target = row['body']


        t0 = time.time()
        results = retriever.get_relevant_documents(query)
        t = time.time() - t0
        retrieval_latency.append(str(t))

        result_content = [result.page_content for result in results]
        # results_content = get_relevant_documents(query, retriever, top_k=5)

        for i, text in enumerate(result_content):
            query_result[i].append(text) 

        if target in result_content:
            retrieval_result.append("Success")
        else:
            retrieval_result.append("Failed")
        # if j>20:
        #     break

    df["retrieval_result"] = retrieval_result 
    df["retrieval_latency"] = retrieval_latency
    for i in range(TOP_N):
        df[f'q{i+1}'] = query_result[i]
        df.to_csv('benchmark_result.csv')
    print(df['retrieval_result'].value_counts())
    print(df['retrieval_result'].value_counts()/ len(df))


