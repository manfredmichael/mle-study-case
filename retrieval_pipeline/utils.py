def get_hybrid_indexes(index_name):
    index_dense = f'{index_name}-dense'
    index_sparse = f'{index_name}-sparse'

    return index_dense, index_sparse