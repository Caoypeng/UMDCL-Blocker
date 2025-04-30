# GiG
import sys

sys.path.append("../")

import os

import numpy as np
import pandas as pd
import itertools

import blocking_utils

import time
from scipy.spatial import distance
from datasets_type import *


def blocks2candidate(left_len, blocks, dataset):
    candidate_set = []
    for block in blocks.keys():
        left_candidate = []
        right_candidate = []
        for i in blocks[block]:
            if i >= left_len:
                right_candidate.append(i - left_len)
            else:
                left_candidate.append(i)
        if dataset in D_type:
            path = os.path.join("../data", dataset, "tables.csv")
            data_df = pd.read_csv(path)
            idx2id = {}
            for i in data_df.index:
                idx2id[i] = data_df.iloc[i]["id"]
            record_set = []
            for i in left_candidate:
                record_set.append(idx2id[i])
            record_set.sort()
            candidate_set.extend(list(itertools.combinations(record_set, 2)))

        else:
            candidate_set.extend(list(itertools.product(left_candidate, right_candidate)))

    candidate_set = list(set(candidate_set))

    candidate_set_df = pd.DataFrame(candidate_set, columns=['ltable_id', 'rtable_id'])
    return candidate_set_df


def load_embeddding(dataset, DR_method, dim):
    if dataset in D_type:
        load_path = os.path.join("../result/embeddings", DR_method, dataset, str(dim) + ".npy")
        embedddings = np.load(load_path)
        return embedddings, embedddings.shape[0]
    elif dataset in D1D2_type:
        load_pathA = os.path.join("../result/embeddings", DR_method, dataset, str(dim) + "_A.npy")
        load_pathB = os.path.join("../result/embeddings", DR_method, dataset, str(dim) + "_B.npy")
        embedddingsA = np.load(load_pathA)
        embedddingsB = np.load(load_pathB)
        embedddings = np.concatenate((embedddingsA, embedddingsB), axis=0)
        return embedddings, embedddingsA.shape[0]


# This is the Abstract Base Class for all Vector Pairing models
class ABCVectorPairing:
    def __init__(self):
        pass

        # Input is an embedding matrix: #tuples x #dimension

    def index(self, embedding_matrix):
        pass

        # Input is an embedding matrix: #tuples x #dimension

    # Output: is a matrix of size #tuples x K where K is application dependent
    def query(self, embedding_matrix):
        pass

    # This is a top-K based blocking strategy


# We index the tuple embeddings from one of the datasets and query the othe
# This is an expensive approach that computes all pair cosine and similarity
# and then extracts top-K neighbors
class ExactTopKVectorPairing(ABCVectorPairing):
    def __init__(self, K, metric="euclidean"):
        super().__init__()
        self.K = K
        self.metric = metric

    # Input is an embedding matrix: #tuples x #dimension
    def index(self, embedding_matrix_for_indexing):
        self.embedding_matrix_for_indexing = embedding_matrix_for_indexing

    # Input is an embedding matrix: #tuples x #dimension
    # Output: is a matrix of size #tuples x K where K is an optional parameter
    # the j-th entry in i-th row corresponds to the top-j-th nearest neighbor for i-th row
    def query(self, embedding_matrix_for_querying, K=None):
        if K is None:
            K = self.K

        # Compute the cosine similarity between two matrices with same number of dimensions
        # E.g. N1 x D and N2 x D, this outputs a similarity matrix of size N1 x N2
        # Note: we pass embedding matrix for querying first and then indexing so that we get
        # top-K neighbors in the indexing matrix
        if self.metric == "cosine":
            all_pair_similarity_matrix = 1 - distance.cdist(embedding_matrix_for_querying,
                                                                   self.embedding_matrix_for_indexing,
                                                                   metric=self.metric)
        elif self.metric == "euclidean":
            all_pair_similarity_matrix = -distance.cdist(embedding_matrix_for_querying,
                                                               self.embedding_matrix_for_indexing,
                                                               metric=self.metric)
        else:
            raise "Use Euclidean distance or cosine similarity"
        # -all_pair_cosine_similarity_matrix is needed to get the max.. use all_pair_cosine_similarity_matrix for min
        topK_indices_each_row = np.argsort(-all_pair_similarity_matrix)[:, :K]
        # you can get the corresponding simlarities via all_pair_cosine_similarity_matrix[index, topK_indices_each_row[index]]

        return topK_indices_each_row


def do_topk1(dataset, DR_method, dim, k):
    print("-" * 10, "top_k", "-" * 10)
    print("dataset：", dataset)
    print("DR method：", DR_method)
    print("dim:", dim)
    print("Top-K param：", "K=", k)
    path = os.path.join("../result/candidate", dataset, "topk", DR_method + "_" + str(dim) + "_" + str(k) + ".csv")


    embeddings, len_A = load_embeddding(dataset, DR_method, dim)

    star_time = time.time()
    topk = ExactTopKVectorPairing(K=k)
    if dataset in D_type:
        idx2id = blocking_utils.idx2id(dataset)

        topk.index(embeddings)

        topK_neighbors = topk.query(embeddings, K=k + 1)
        end_time = time.time()


        topK_neighbors = topK_neighbors[:, 1:]



        candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)
        candidate_set = []
        for i in candidate_set_df.index:
            l = candidate_set_df.iloc[i]["ltable_id"]
            r = candidate_set_df.iloc[i]["rtable_id"]
            l_id = idx2id[l]
            r_id = idx2id[r]
            if l_id < r_id:
                candidate_set.append((l_id, r_id))
            elif l_id > r_id:
                n = l_id
                l_id = r_id
                r_id = n
                candidate_set.append((l_id, r_id))
        candidate_set = list(set(candidate_set))
        candidate_set_df = pd.DataFrame(candidate_set, columns=['ltable_id', 'rtable_id'])
    elif dataset in D1D2_type:

        topk.index(embeddings[len_A:])

        topK_neighbors = topk.query(embeddings[0:len_A], K=k)
        end_time = time.time()
        candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)

    path = os.path.join("../result/candidate", dataset, "topk", DR_method + "_" + str(dim) + "_" + str(k) + ".csv")
    candidate_set_df.to_csv(path, index=False)
    print(len(candidate_set_df))
    print("result path：", path)

    # path = os.path.join("candidate",args.dataset,"kmeans",args.DR_method+"_"+str(args.dim)+"_"+args.num_clusters+".csv")
    path1 = os.path.join("../data", dataset, "matches.csv")
    golden_df = pd.read_csv(path1)
    if dataset in D1D2_type:
        pathA = os.path.join("../data", dataset, "tableA.csv")
        pathB = os.path.join("../data", dataset, "tableB.csv")
        A_df = pd.read_csv(pathA, encoding="gbk")
        B_df = pd.read_csv(pathB, encoding="gbk")
        statistics_dict = blocking_utils.compute_blocking_statistics_F1(candidate_set_df, golden_df, A_df, B_df)
    elif dataset in D_type:
        path = os.path.join("../data", dataset, "tables.csv")
        data_df = pd.read_csv(path)
        statistics_dict = blocking_utils.compute_blocking_statistics_F1(candidate_set_df, golden_df, data_df, data_df)
    t = end_time - star_time
    print(t)
    print(statistics_dict)
    # save result
    result_path = os.path.join("../result/output", dataset, "topk", dataset + "_topk_test_result.csv")
    result = pd.DataFrame({"cluster_method": ["topk"], "DR_method": [DR_method], "dim": [dim], "K": [k], "PC": [statistics_dict.get('PC')], "RR": [statistics_dict.get('RR')], "F-alpha": [statistics_dict.get('F-alpha')], "time": [t]})
    if not os.path.isfile(result_path):
        result.to_csv(result_path, index=False)
    else:
        result.to_csv(result_path, mode='a', header=False, index=False)
    return statistics_dict


def load_embeddings(dataset, DR_method, dim):
    if dataset in D_type:
        query_path = os.path.join("../result/embeddings", DR_method, dataset, str(dim) + ".npy")
        index_path = os.path.join("../result/embeddings", DR_method, dataset, str(dim) + ".npy")
    elif dataset in D1D2_type:
        query_path = os.path.join("../result/embeddings", DR_method, dataset, str(dim) + "_A.npy")
        index_path = os.path.join("../result/embeddings", DR_method, dataset, str(dim) + "_B.npy")


    query_data = np.load(query_path)
    query_len = query_data.shape[0]

    index_data = np.load(index_path)
    index_len = index_data.shape[0]
    d = query_data.shape[1]

    return query_data, index_data, query_len, index_len, d


def blocks2candidates(blocks, query_len, dataset):
    candidates = []
    if dataset in D_type:
        idx2id = blocking_utils.idx2id(dataset)
        for i in range(query_len):
            l_id = idx2id[i]
            for j in range(len(blocks[i])):
                r_id = idx2id[blocks[i][j]]
                if l_id == r_id:
                    continue
                elif l_id < r_id:
                    candidates.append((l_id, r_id))
                else:
                    candidates.append((r_id, l_id))

        candidates = list(set(candidates))
    elif dataset in D1D2_type:
        for i in range(query_len):
            l_id = i
            for j in range(len(blocks[i])):
                r_id = blocks[i][j]
                candidates.append((l_id, r_id))
    return candidates


def evaluate(dataset, candidates, query_len, index_len):
    match_path = os.path.join("../data", dataset, "matches.csv")
    matches = pd.read_csv(match_path)
    merged = pd.merge(candidates, matches, on=['ltable_id', 'rtable_id'])
    PC = len(merged) / len(matches)
    if dataset in D_type:
        RR = 1.0 - (2 * len(candidates) / int(query_len * (index_len - 1)))
    elif dataset in D1D2_type:
        RR = 1.0 - (len(candidates) / int(query_len * index_len))
    F_alpha = 2 * PC * RR / max(PC + RR, 1)
    return PC, RR, F_alpha


def do_topk(dataset, DR_method, dim, k, metric="euclidean"):
    print("cluster method: topk", "dataset: ", dataset, "DR_method: ", DR_method, "dim: ", dim, "k: ", k)

    query_data, index_data, query_len, index_len, d = load_embeddings(dataset, DR_method, dim)

    start_time = time.time()
    topk = ExactTopKVectorPairing(K=k, metric=metric)  # cosine
    topk.index(index_data)
    if dataset in D_type:
        blocks = topk.query(query_data, K=k + 1)
    elif dataset in D1D2_type:
        blocks = topk.query(query_data, K=k)
    end_time = time.time()
    t = end_time - start_time
    # print("time: ", end_time - start_time)
    # print(blocks.shape)

    candidates = blocks2candidates(blocks, query_len, dataset)
    candidates = pd.DataFrame(candidates, columns=['ltable_id', 'rtable_id'])

    candidate_path = os.path.join("../result/candidate", dataset, "topk",
                                  DR_method + "_" + str(dim) + "_" + str(k) + ".csv")
    if not os.path.exists(os.path.join("../result/candidate", dataset, "topk")):
        os.makedirs(os.path.join("../result/candidate", dataset, "topk"))
    candidates.to_csv(candidate_path, index=False)
    PC, RR, F_alpha = evaluate(dataset, candidates, query_len, index_len)
    print("PC: ", PC, "RR: ", RR, "F_alpha: ", F_alpha, "time: ", t)

    if not os.path.exists(os.path.join("../result/output", dataset, "topk")):
        os.makedirs(os.path.join("../result/output", dataset, "topk"))

    result_path = os.path.join("../result/output", dataset, "topk", dataset + f"_topk_{metric}_result.csv")
    result = pd.DataFrame({"cluster_method": ['topk'], "DR_method": [DR_method], "dim": [dim], "K": [k],
                           "PC": [PC], "RR": [RR],
                           "F-alpha": [F_alpha], "time": [t]})
    if not os.path.isfile(result_path):
        result.to_csv(result_path, index=False)
    else:
        df = pd.read_csv(result_path)
        temp_df = df[df["DR_method"] == DR_method]
        if len(temp_df) >= 10:
            df = df[df["DR_method"] != DR_method]
            df.to_csv(result_path, index=False)
        result.to_csv(result_path, mode='a', header=False, index=False)


if __name__ == '__main__':
    # 
    datasets = ["Amazon-Google2","Abt-Buy","X1","X2","cora","shoes_small","cameras_small","watches_small","computers_small"]
    Ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    DR_methods = ["/home/cyp/uB-CL-main-colbertgate/result/embeddings/output"]  # DR METHODS list
    # do_topk(dataset="X1",DR_method="SBERT",dim=12,k=10)
    for dataset in datasets:
        for DR_method in DR_methods:
            for K in Ks:
                do_topk(dataset=dataset, DR_method=DR_method, dim=12, k=K, metric="cosine")
    # for K in Ks:
    #     do_topk(dataset="Amazon-Google2", DR_method="SBERT", dim=12, k=K)
    # do_topk(dataset="Abt-Buy", DR_method="SBERT", dim=12, k=10)

