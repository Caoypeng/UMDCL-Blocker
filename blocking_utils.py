import os.path

import pandas as pd
#基于top-k生成候选集
def topK_neighbors_to_candidate_set(topK_neighbors):
    #We create a data frame corresponding to topK neighbors.
    # We are given a 2D matrix of the form 1: [a1, a2, a3], 2: [b1, b2, b3]
    # where a1, a2, a3 are the top-3 neighbors for tuple 1 and so on.
    # We will now create a two column DF fo the form (1, a1), (1, a2), (1, a3), (2, b1), (2, b2), (2, b3)
    topK_df = pd.DataFrame(topK_neighbors)
    topK_df["ltable_id"] = topK_df.index
    melted_df = pd.melt(topK_df, id_vars=["ltable_id"])
    melted_df["rtable_id"] = melted_df["value"]
    candidate_set_df = melted_df[["ltable_id", "rtable_id"]]
    return candidate_set_df


#This accepts four inputs:
# data frames for candidate set and ground truth matches
# left and right data frames




def compute_blocking_statistics_F1(candidate_set_df, golden_df, left_df, right_df):
    #计算PC和PQ的F调和平均
    #Now we have two data frames with two columns ltable_id and rtable_id
    # If we do an equi-join of these two data frames, we will get the matches that were in the top-K
    merged_df = pd.merge(candidate_set_df, golden_df, on=['ltable_id', 'rtable_id'])
    left_num_tuples = len(left_df)
    right_num_tuples = len(right_df)
    PC = len(merged_df) / len(golden_df)
    if left_num_tuples == right_num_tuples:
        RR = 1.0 - (2*len(candidate_set_df) / int(left_num_tuples * (right_num_tuples-1)))
    else:
        RR = 1.0 - (len(candidate_set_df) / int(left_num_tuples * right_num_tuples))


    CSSR = len(candidate_set_df) / (left_num_tuples * right_num_tuples)
    #PQ = len(merged_df) / len(candidate_set_df)
    PQ = 0
    F_alpha = 2 * PC * RR / max(PC + RR, 1)
    # F_1 = 2 * PC * PQ / max(PC + PQ, 1)
    F_1 = 0
    statistics_dict = {
        "left_num_tuples": left_num_tuples,
        "right_num_tuples": right_num_tuples,
        "PC": PC,
        "RR": RR,
        "F-alpha":F_alpha,
        "F-1":F_1,
        "PQ":PQ
        }
    return statistics_dict



#This function is useful when you download the preprocessed data from DeepMatcher dataset
# and want to convert to matches format.
#It loads the train/valid/test files, filters the duplicates,
# and saves them to a new file called matches.csv
#这个方法是想把train set、valid set、test set中的匹配的元组对找出来
def process_files(folder_root):
    df1 = pd.read_csv(folder_root + "/train.csv")
    df2 = pd.read_csv(folder_root + "/valid.csv")
    df3 = pd.read_csv(folder_root + "/test.csv")

    df1 = df1[df1["label"] == 1]
    df2 = df2[df2["label"] == 1]
    df3 = df3[df3["label"] == 1]

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df[["ltable_id","rtable_id"]].to_csv(folder_root + "/matches.csv", header=True, index=False)

def idx2id(dataset):
    path = os.path.join("../data",dataset,"tables.csv")
    idx2id = {}
    try:
        data_df = pd.read_csv(path)
    except:
        # 如果报错，说明实在demo中读取的，demo在 ann-benchmarks-main/demo中
        path = os.path.join("..", path)
        data_df = pd.read_csv(path)
    # 建立一个字典，idx->id
    for i in data_df.index:
        idx2id[i] = data_df.iloc[i]["id"]

    return idx2id

def insert_pd(path,insert_df):
    if os.path.exists(path):
        df1 = pd.read_csv(path)
        output_df = pd.concat([df1, insert_df], ignore_index=True)
        output_df.to_csv(path, index=False)
        print("插入成功")
    else:
        insert_df.to_csv(path, index=False)
        print("目标路径不存在csv文件，已自动创建")



def addSchemaInf(data_df,cols):
    #保存模式信息的序列化方法 [CAL]name[VAL]xuyang....
    if "id" in cols:
        cols.remove("id")
    tuple_id_list = []
    tuple_body_list = []
    for idx in data_df.index:
        tuple = data_df.iloc[idx]
        tuple_id = tuple["id"]
        tuple_body = ""
        for att in cols:
            val = tuple[att]
            tuple_body = tuple_body+"[COl]"+att+"[VAL]"+val+" "
        tuple_id_list.append(tuple_id)
        tuple_body_list.append(tuple_body)
    data_dic = {"id":tuple_id_list,"_merged_text":tuple_body_list}
    data_df = pd.DataFrame(data_dic)
    return data_df


def timeit(func):
    def wrapper(*args,**kwargs):
        import time
        start = time.time()
        func(*args,**kwargs)
        end = time.time()
        print("耗时：",end-start)
    return wrapper