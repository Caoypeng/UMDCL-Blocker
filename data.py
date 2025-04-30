import os
from torch.utils.data import Dataset
from arguments import DataArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
import datasets
import random
from gensim.models import Word2Vec
import logging
from transformers import DataCollator
from dataclasses import dataclass
import faiss
import numpy as np
logger = logging.getLogger(__name__)
logging.getLogger('gensim').setLevel(logging.ERROR)

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords

from threading import Lock

stopwords = set(stopwords.words('english'))

class TrainDatasetForCL(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if not args.hard_negative:
            data_files = {}
            fn = "/home/cyp/uB-CL-main/uB-CL-main/data/data1/all_data_merge.csv"  # Assuming the dataset is in the "data" folder
            # 这个all_data_merge是原始记录加会议数据对吧 ok
             # dataset is the filename
            content = []
            with open(fn) as fin:
                for line in fin:
                    LL = line.split('\t')
                    for entry in LL:
                        content.append(entry)
                # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                token_pattern=r'(?u)\S+',  # 匹配包含括号和连字符的标记
                lowercase=False,                 # 保留原始大小写
                stop_words=None                # 不使用停用词过滤
            ).fit(content)
            self.vocab = vectorizer.vocabulary_
            self.idf = vectorizer.idf_

            if args.train_file is not None:
                data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            if extension == "csv":
                self.datasets = datasets.load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter=args.delimiter)
            else:
                self.datasets = datasets.load_dataset(extension, data_files=data_files, cache_dir="./data/")
            self.datasets.shuffle(seed=42)
            self.tokenizer = tokenizer
            self.args = args

            self.column_names = self.datasets["train"].column_names
            if len(self.column_names) == 1:
                self.sent1 = self.column_names[0]
                self.sent2 = self.column_names[0]
            elif len(self.column_names) == 2:
                self.sent1 = self.column_names[0]
                self.sent2 = self.column_names[1]
            else:
                raise NotImplementedError("The dataset should have only one or two columns")

            self.total_len = len(self.datasets["train"])
            # print(self.total_len)
            if self.args.delete_word:
                logger.info("Delete words with probability %f", self.args.delete_word_probability)
            if self.args.swap_word:
                logger.info("Swap words with probability %f", self.args.swap_word_probability)
            if self.args.replace_word:
                logger.info("Replace words with probability %f", self.args.replace_word_probability)
                self.FAISS_AVAILABLE = True if self.args.hnsw_index else False
                if self.FAISS_AVAILABLE:
                    if not os.path.exists(self.args.hnsw_index):
                        raise ValueError("hnws index is an invalid path!")
                    self.INDEX = faiss.read_index(self.args.hnsw_index)
                if not self.args.word2vec_model or not os.path.exists(self.args.word2vec_model):
                    raise ValueError("word2vec model is an invalid path!")
                self.WORD2VEC_MODEL = Word2Vec.load(self.args.word2vec_model)

        else:
            self.dataset = datasets.load_dataset("json", data_files=args.train_file, split="train", cache_dir="./data/")
            self.tokenizer = tokenizer
            self.args = args
            self.total_len = len(self.dataset)
    def get_len(self, word):
        """Return the sentence_piece length of a token.
        """
        length = len(self.tokenizer.tokenize(word))
        return length
    def summarize_text(self, text, topk=126, max_bert_len=126):
        """
        对给定文本进行摘要操作，保留加权后的关键token。
        新增规则：
        1. 位置权重：越靠前的token权重越高（线性衰减）
        2. Hashtag权重：以#开头的token权重翻倍
        """
        tokens = text.split()

        if self.get_len(text) <= max_bert_len:
            return text
        
        cnt = Counter()
        total_tokens = len(tokens)

        # 计算每个 token 的 tf-idf 权重（仅考虑非停用词且在词汇表中的 token）
        for idx, token in enumerate(tokens):
            if token in self.vocab:
                score = self.idf[self.vocab[token]]


                # # 规则1：位置权重——越靠前的 token 权重越高
                # # 这里采用前50%的 token 乘以 10.0
                if idx < total_tokens * 0.5:
                    score *= 5.0

                # 规则2：数字权重——如果 token 是数字，则进一步提高权重
                # if token.isdigit():
                #     score *= 2.0
                cnt[token] += score
                        
        # 获取 tf-idf 权重最高的前 topk 个 token，形成候选 token 集合
        topk_tokens = {token for token, _ in cnt.most_common(topk)}

        # 根据 BERT 分词长度控制摘要的总长度
        summary_tokens = []
        current_length = 0
        for token in tokens:
            # 如果 token 在候选集合中，则考虑加入摘要
            if token in topk_tokens:
                token_len = self.get_len(token)  # 获取 token 的 BERT 分词长度
                if current_length + token_len <= max_bert_len:
                    summary_tokens.append(token)
                    current_length += token_len
                # 如果累计长度达到或超过 max_bert_len，则停止添加
                if current_length >= max_bert_len:
                    break

        return ' '.join(summary_tokens).strip()
    # def summarize_text(self, text, topk=126, max_bert_len=126):
    #     """
    #     对给定文本进行摘要操作，在保留原始顺序的前提下优先保留高IDF值的token。
    #     参数:
    #         topk: 用于候选token选择的数量阈值。
    #         max_bert_len: 摘要允许的最大BERT分词长度。
    #     """
    #     tokens = text.split()
    #     if self.get_len(text) <= max_bert_len:
    #         return text

    #     # 预处理：过滤停用词和不在词汇表的token，并记录IDF和长度
    #     processed = []
    #     for token in tokens:
    #         if token in self.vocab:
    #             token_idf = self.idf[self.vocab[token]]
    #             token_len = self.get_len(token)
    #             processed.append((token, token_idf, token_len, len(processed)))  # 保留原始索引
    #         else:
    #             processed.append((token, -1, self.get_len(token), len(processed)))    # 标记无效token

    #     # 筛选候选token (IDF最高的topk个，按原始顺序排列)
    #     valid_candidates = [item for item in processed if item[1] != -1]
    #     valid_candidates.sort(key=lambda x: (-x[1], x[3]))  # 按IDF降序，原始索引升序排序
    #     topk_candidates = valid_candidates[:topk]
    #     topk_candidates.sort(key=lambda x: x[3])  # 恢复原始顺序

    #     # 动态选择token并控制长度
    #     selected = []
    #     current_length = 0
    #     for token_info in topk_candidates:
    #         token, token_idf, token_len, _ = token_info
            
    #         if current_length + token_len <= max_bert_len:
    #             selected.append(token_info)
    #             current_length += token_len
    #         else:
    #             # 计算需要腾出的空间
    #             needed_space = current_length + token_len - max_bert_len
                
    #             # 收集可删除的候选(按IDF升序，位置升序)
    #             sorted_candidates = sorted(
    #                 [(idx, t[1], t[2]) for idx, t in enumerate(selected)],
    #                 key=lambda x: (x[1], x[0])
    #             )
                
    #             # 尝试删除低价值token
    #             removed_length = 0
    #             removed_indices = []
    #             for candidate in sorted_candidates:
    #                 if removed_length >= needed_space:
    #                     break
                    
    #                 if (current_length - candidate[2] + token_len) <= max_bert_len:
    #                     removed_length += candidate[2]
    #                     removed_indices.append(candidate[0])
                
    #             # 执行删除操作
    #             if removed_length > 0:
    #                 for idx in sorted(removed_indices, reverse=True):
    #                     current_length -= selected[idx][2]
    #                     del selected[idx]
                    
    #                 # 添加当前token
    #                 if current_length + token_len <= max_bert_len:
    #                     selected.append(token_info)
    #                     current_length += token_len

    #     # 按原始顺序提取结果
    #     selected.sort(key=lambda x: x[3])  # 按原始索引排序
    #     summary_tokens = [item[0] for item in selected]
        
    #     return ' '.join(summary_tokens).strip()
    # def summarize_text(self, text, max_len=126):
    #     """对给定的文本进行摘要操作，保留最高的 tf-idf tokens"""
    #     cnt = Counter()
    #     tokens = text.split(' ')
    #     res = ''

    #     # 计算每个 token 的 tf-idf 权重
    #     for token in tokens:
    #         if token not in stopwords:
    #             if token in self.vocab:
    #                 cnt[token] += self.idf[self.vocab[token]]
                    
    #     # 获取 tf-idf 排名前 max_len 的 tokens
    #     token_cnt = Counter(tokens)
    #     subset = Counter()
    #     for token in set(token_cnt.keys()):
    #         subset[token] = cnt[token]
    #     subset = subset.most_common(max_len)
    #     # 保留 top-k 的 token
    #     topk_tokens_copy = set([])  # topk tokens that we want to keep
    #     total_len = 2  # 用于计算总长度，避免超过 max_len
    #     for word, _ in subset:
    #         bert_len = self.get_len(word)
    #         if total_len + bert_len > max_len:
    #             break
    #         total_len += bert_len
    #         topk_tokens_copy.add(word)

    #     # 生成摘要文本
    #     for token in tokens:
    #         if token in topk_tokens_copy:
    #             res += token + ' '
    #             topk_tokens_copy.remove(token)
        
    #     return res.strip()  # 返回摘要后的文本      subset[token] = cnt[token]
  
    def __getitem__(self, item):
        query = self.dataset[item]["query"]
        pos = random.choice(self.dataset[item]["pos"])
        neg = self.dataset[item]["neg"]
        data = [query] + [pos] + neg[:(self.args.grouped_size-1)]
        # data = [query] + [pos] + neg[:10]
        tokenizer_dataset = self.tokenizer(
            data,
            padding="max_length" if self.args.pad_to_max_length else False,
            truncation=True,
            max_length=self.args.max_seq_length,
        )
        return tokenizer_dataset


    def __len__(self):
        return self.total_len
    
    
    def prepare_features(self, examples):
        total = len(examples[self.sent1])
        summarized_sentences = []
        # Avoid "None" fields 
        for idx in range(total):
            if examples[self.sent1][idx] is None:
                examples[self.sent1][idx] = " "
            if examples[self.sent2][idx] is None:
                examples[self.sent2][idx] = " "
            # print("************************")
            # print(examples[self.sent1][idx])
            # print(self.get_len(examples[self.sent1][idx]))
            summarized_text = self.summarize_text(examples[self.sent1][idx])
            # print(summarized_text)
            # print(self.get_len(summarized_text))
            summarized_sentences.append(summarized_text)
        examples[self.sent1] = summarized_sentences
        new_sentences2 = []
        for idx in range(total):
            temp = examples[self.sent2][idx]
            if self.args.delete_word:
                if self.args.delete_word_probability is None:
                    raise "The probability must be provided and the range is [0, 1]."
                elif not 0 <= self.args.delete_word_probability <= 1.0:
                    raise "The probability has to be greater than or equal to 0 and less than or equal to 1."
                temp = self.delete_words(temp, p=self.args.delete_word_probability)
            if self.args.swap_word:
                if self.args.swap_word_probability is None:
                    raise "The probability must be provided and the range is [0, 1]."
                elif not 0 <= self.args.swap_word_probability <= 1.0:
                    raise "The probability has to be greater than or equal to 0 and less than or equal to 1."
                temp = self.swap_words(temp, p=self.args.swap_word_probability)
            if self.args.replace_word:
                if self.args.replace_word_probability is None:
                    raise "The probability must be provided and the range is [0, 1]."
                elif not 0 <= self.args.replace_word_probability <= 1.0:
                    raise "The probability has to be greater than or equal to 0 and less than or equal to 1."
                temp = self.replace_words(temp, p=self.args.replace_word_probability, is_random=False)
            # print(temp)
            # print(self.get_len(temp) )
            # print("***************888")
            temp=self.summarize_text(temp)
            # print(temp)
            # print(self.get_len(temp) )

            # print("*1111111111111111111")
            # summarized_text2 = self.summarize_text(temp)
            new_sentences2.append(temp)
        sentences = examples[self.sent1] + new_sentences2






        sent_features = self.tokenizer(
            sentences,
            max_length=self.args.max_seq_length,
            truncation=True,
            padding="max_length" if self.args.pad_to_max_length else False,
        )
        features = {}
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        return features

    # DA
    def delete_words(self, sentence, p=0.1):
        words = self.tokenizer.tokenize(sentence)
        words = [word for word in words if random.random() > p]
        return " ".join(words)

    def swap_words(self, sentence, p=0.1):
        words = self.tokenizer.tokenize(sentence)
        for i in range(1, len(words)):
            if random.random() < p:
                words[i], words[i-1] = words[i-1], words[i]
        return " ".join(words)

    def replace_words(self, sentence, p=0.1, is_random=False):
        words = self.tokenizer.tokenize(sentence)

        for i in range(len(words)):
            if random.random() < p:
                if not is_random:
                    try:
                        if not self.FAISS_AVAILABLE:
                            sim_words = self.WORD2VEC_MODEL.wv.most_similar(words[i].lower(), topn=5)
                            words[i] = sim_words[random.randint(0, 4)][0]
                        else:
                                query_word = words[i].lower()
                                if query_word in self.WORD2VEC_MODEL.wv:
                                    query_vector = np.array([self.WORD2VEC_MODEL.wv[query_word]]).astype('float32')
                                    faiss.normalize_L2(query_vector)
                                    distances, indices = self.INDEX.search(query_vector, 6)
                                    words[i] = self.WORD2VEC_MODEL.wv.index_to_key[indices[0][random.randint(1, 5)]]
                                else:
                                    words[i] = self.WORD2VEC_MODEL.wv.index_to_key[
                                        random.randint(0, len(self.WORD2VEC_MODEL.wv.index_to_key) - 1)]
                    except Exception as e:
                        print(e)
                        words[i] = self.WORD2VEC_MODEL.wv.index_to_key[random.randint(0, len(self.WORD2VEC_MODEL.wv.index_to_key) - 1)]
                else:
                    words[i] = self.WORD2VEC_MODEL.wv.index_to_key[random.randint(0, len(self.WORD2VEC_MODEL.wv.index_to_key) - 1)]
        return " ".join(words)


@dataclass
class CLCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding = True
    max_length = None
    pad_to_multiple_of = None

    def __call__(self, features):
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )


        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for
                 k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch