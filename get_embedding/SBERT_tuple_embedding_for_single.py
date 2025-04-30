#GiG
import sys
sys.path.append("../")
import argparse
import os
import cols_to_block
import numpy as np
import pandas as pd
from pathlib import Path
from tuple_embedding_models import  SentenceBertTupleEmbedding,SentenceBert_layer
import blocking_utils
import time


def prepared_dataset(dataset,cols_to_block):

    path = os.path.join("../data", dataset, "tables.csv")
    data_df = pd.read_csv(path)
    if "id" not in cols_to_block:
        cols_to_block.append("id")
    cols_to_block_without_id = [col for col in cols_to_block if col != "id"]
    # Check if all required columns are in left_df
    check = all([col in data_df.columns for col in cols_to_block])
    if not check:
        raise Exception("Not all columns in cols_to_block are present in the left dataset")
    data_df = data_df[cols_to_block]
    data_df.fillna(' ', inplace=True)
    data_df = data_df.astype(str)
    data_df["_merged_text"] = data_df[cols_to_block_without_id].agg(' '.join, axis=1)
    data_df = data_df.drop(columns=cols_to_block_without_id)
    return data_df



def get_embeeding_for_single(dataset,layer, model, model_name):
    from cols_to_block import cols_to_block
    cols_to_block = cols_to_block[dataset]
    print("dataset：",dataset)
    print("block attr：",cols_to_block)
    print("layer number：",layer)
    print("-"*10,"dataset preprocess","-"*10)
    data_df = prepared_dataset(dataset,cols_to_block)
    print("-"*10,"init pre-train sentence transformer","-"*10)
    tuple_embedding_model = SentenceBert_layer(layer, model)
    print("-"*10,"generate tuple embedding","-"*10)
    star_time = time.time()
    tuple_embeddings = tuple_embedding_model.get_tuple_embedding(data_df["_merged_text"].values.tolist())
    end_time = time.time()
    if not os.path.exists(os.path.join("../result/embeddings", model_name, dataset)):
        os.makedirs(os.path.join("../result/embeddings", model_name, dataset))
    save_path = os.path.join("../result/embeddings", model_name, dataset, str(layer)+".npy")
    print("tupple embeddings result：", save_path)
    np.save(save_path, tuple_embeddings)

    if not os.path.exists(os.path.join("../result","time","DR",model_name)):
        os.makedirs(os.path.join("../result","time","DR",model_name))
    path = os.path.join("../result","time","DR",model_name,"DR_cost.csv")
    time_df = pd.DataFrame({"dataset":dataset,"DR_method":model_name,"layer":layer,"time":end_time-star_time},index=[0])
    if os.path.exists(path):
        df1 = pd.read_csv(path)
        output_df = pd.concat([df1, time_df], ignore_index=True)
        output_df.to_csv(path, index=False)
    else:
        time_df.to_csv(path, index=False)



if __name__ == '__main__':
    emdbedding_model = ["/home/cyp/uB-CL-main-colbertgate/output"]  # path list

    count = 1
    for embedding in emdbedding_model:
        for dataset in ["X1", "X2", "cora", "shoes_small", "cameras_small", "watches_small", "computers_small"]:
            get_embeeding_for_single(dataset=dataset, layer=12,
                                     model=embedding,
                                     model_name=embedding.split("/")[-1])
        count += 1


