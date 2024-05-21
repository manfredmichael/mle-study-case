import argparse
import pandas as pd

def find_reciprocal_rank(target, row, u, k): 
    for i in range(k):
        q = row['q{}'.format(i+1)]
        if target == q:
            print(1/(i+1))
            return 1/(i+1) 
    return 0

def main(filename, k):
    df = pd.read_csv(filename)
    u = len(df)

    sum_ = 0
    for _, row in df.iterrows():
        target = row['body']
        reciprocal_rank = find_reciprocal_rank(target, row, u, k)
        sum_ += reciprocal_rank
    mrr = sum_ / u

    print('U:', u)
    print('MRR: ', mrr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename') 
    parser.add_argument('-k', type=int) 
    args = parser.parse_args()

    main(filename=args.filename, k=args.k)