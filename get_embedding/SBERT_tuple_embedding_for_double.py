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
import time



def prepared_dataset(dataset,cols_to_block):

    left_path = os.path.join("../data", dataset, "tableA.csv")
    right_path = os.path.join("../data", dataset, "tableB.csv")
    left_df = pd.read_csv(left_path,encoding="gbk")
    right_df = pd.read_csv(right_path,encoding="gbk")
    if "id" not in cols_to_block:
        cols_to_block.append("id")
    cols_to_block_without_id = [col for col in cols_to_block if col != "id"]
    # Check if all required columns are in left_df
    check = all([col in left_df.columns for col in cols_to_block])
    if not check:
        raise Exception("Not all columns in cols_to_block are present in the left dataset")
    # Check if all required columns are in left_df
    check = all([col in right_df.columns for col in cols_to_block])
    if not check:
        raise Exception("Not all columns in cols_to_block are present in the left dataset")
    left_df = left_df[cols_to_block]
    right_df = right_df[cols_to_block]
    left_df.fillna(' ', inplace=True)
    right_df.fillna(' ', inplace=True)
    left_df = left_df.astype(str)
    right_df = right_df.astype(str)
    left_df["_merged_text"] = left_df[cols_to_block_without_id].agg(' '.join, axis=1)
    right_df["_merged_text"] = right_df[cols_to_block_without_id].agg(' '.join, axis=1)
    left_df = left_df.drop(columns=cols_to_block_without_id)
    right_df = right_df.drop(columns=cols_to_block_without_id)
    return left_df,right_df



def get_embeeding_for_double(dataset,layer, model, model_name):
    from cols_to_block import cols_to_block
    cols_to_block = cols_to_block[dataset]
    print("dataset：",dataset)
    print("block attr：",cols_to_block)
    print("layer number ：",layer)
    print("-"*10,"dataset preprocess","-"*10)
    left_df,right_df = prepared_dataset(dataset,cols_to_block)
    print("-"*10,"init pre-train sentence transformer","-"*10)
    tuple_embedding_model = SentenceBert_layer(layer, model_name=model)
    print("-"*10,"generate tuple embedding","-"*10)
    star_time = time.time()
    left_tuple_embeddings = tuple_embedding_model.get_tuple_embedding(left_df["_merged_text"].values.tolist())
    right_tuple_embeddings = tuple_embedding_model.get_tuple_embedding(right_df["_merged_text"].values.tolist())
    end_time = time.time()

    if not os.path.exists(os.path.join("../result/embeddings", model_name, dataset)):
        os.makedirs(os.path.join("../result/embeddings", model_name, dataset))
    left_save_path = os.path.join("../result/embeddings", model_name, dataset, str(layer)+"_A.npy")
    right_save_path = os.path.join("../result/embeddings", model_name, dataset,str(layer)+"_B.npy")
    np.save(left_save_path, left_tuple_embeddings)
    np.save(right_save_path,right_tuple_embeddings)
    print("tupple embeddings result：",left_save_path,"    ",right_save_path)


    path = os.path.join("../result","time","DR",model_name,"DR_cost.csv")
    if not os.path.exists(os.path.join("../result","time","DR",model_name)):
        os.makedirs(os.path.join("../result","time","DR",model_name))
    time_df = pd.DataFrame({"dataset":dataset,"DR_method":model_name,"layer":layer,"time":end_time-star_time},index=[0])
    if os.path.exists(path):
        df1 = pd.read_csv(path)
        output_df = pd.concat([df1, time_df], ignore_index=True)
        output_df.to_csv(path, index=False)
    else:
        time_df.to_csv(path, index=False)



if __name__ == '__main__':
    emdbedding_model = ["/home/cyp/uB-CL-main-colbertgate/output"] # path list
    count = 1
    for embedding in emdbedding_model:
        # ,"Personal Identity Deduplication"
        for dataset in ["Abt-Buy", "Amazon-Google2"]:
            get_embeeding_for_double(dataset=dataset, layer=12,
                                     model=embedding,
                                     model_name=embedding.split("/")[-1])
        count += 1





