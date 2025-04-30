import torch
import dl_models
from configurations import *
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel, BartModel
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import TruncatedSVD
from torch import Tensor, device
import gensim.models as g
import tqdm
from collections import Counter
import codecs
import os
from scipy.sparse import csr_matrix

# import fasttext as fastText
import random
# from torchtext.data import get_tokenizer
from tqdm import tqdm


def mean_pooling(model_output, attention_mask,layer):
    token_embeddings = model_output.hidden_states[layer] #First element of model_output contains all token embeddings  提取第几层的hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # 将attention_mask扩展到和token_embeddings一样的维度，即0为padding的地方，对应的token_embeddings为0
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # 先计算token与attention_mask的乘积，然后除以attention_mask的和，即求平均值

def bart_mean_pooling(model_output, attention_mask,layer, encoder_or_decoder):
    if encoder_or_decoder == "encoder":
        token_embeddings = model_output.encoder_hidden_states[layer] #First element of model_output contains all token embeddings  提取第几层的hidden_state
    else:
        token_embeddings = model_output.decoder_hidden_states[layer] #First element of model_output contains all token embeddings  提取第几层的hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # 将attention_mask扩展到和token_embeddings一样的维度，即0为padding的地方，对应的token_embeddings为0
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # 先计算token与attention_mask的乘积，然后除以attention_mask的和，即求平均值

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def generate_synthetic_training_data(list_of_tuples, synth_tuples_per_tuple=5,
                                     pos_to_neg_ratio=1, max_perturbation=0.4):
    num_positives_per_tuple = synth_tuples_per_tuple
    num_negatives_per_tuple = synth_tuples_per_tuple * pos_to_neg_ratio
    num_tuples = len(list_of_tuples)
    total_number_of_elems = num_tuples * (num_positives_per_tuple + num_negatives_per_tuple)

    # We create three lists containing T, T' and L respectively
    # We use the following format: first num_tuples * num_positives_per_tuple correspond to T
    # and the remaining correspond to T'
    left_tuple_list = [None for _ in range(total_number_of_elems)]
    right_tuple_list = [None for _ in range(total_number_of_elems)]
    label_list = [0 for _ in range(total_number_of_elems)]

    random.seed(RANDOM_SEED)

    tokenizer = get_tokenizer("basic_english")
    for index in range(len(list_of_tuples)):
        tokenized_tuple = tokenizer(list_of_tuples[index])
        max_tokens_to_remove = int(len(tokenized_tuple) * max_perturbation)

        training_data_index = index * (num_positives_per_tuple + num_negatives_per_tuple)

        # Create num_positives_per_tuple tuple pairs with positive label
        for temp_index in range(num_positives_per_tuple):
            tokenized_tuple_copy = tokenized_tuple[:]

            # If the tuple has 10 words and max_tokens_to_remove is 0.5, then we can remove at most 5 words
            # we choose a random number between 0 and 5.
            # suppose it is 3. Then we randomly remove 3 words
            num_tokens_to_remove = random.randint(0, max_tokens_to_remove)
            for _ in range(num_tokens_to_remove):
                # randint is inclusive. so randint(0, 5) can return 5 also
                tokenized_tuple_copy.pop(random.randint(0, len(tokenized_tuple_copy) - 1))

            left_tuple_list[training_data_index] = list_of_tuples[index]
            right_tuple_list[training_data_index] = ' '.join(tokenized_tuple_copy)
            label_list[training_data_index] = 1
            training_data_index += 1

        for temp_index in range(num_negatives_per_tuple):
            left_tuple_list[training_data_index] = list_of_tuples[index]
            right_tuple_list[training_data_index] = random.choice(list_of_tuples)
            label_list[training_data_index] = 0
            training_data_index += 1
    return left_tuple_list, right_tuple_list, label_list


#This is the Abstract Base Class(抽象基类) for all Tuple Embedding models
class ABCTupleEmbedding:
    def __init__(self):
        pass 

    #This function is used as a preprocessing step 
    # this could be used to compute frequencies, train a DL model etc
    #负责计算每个词的频率(SIF)，训练模型、导入预训练的模型等
    def preprocess(self, list_of_tuples):
        pass 

    #This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        pass 

    #This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        pass


#FAST TEXT + Aaverage DeepBlocking 源码
class AverageEmbedding(ABCTupleEmbedding):
    def __init__(self):
        super().__init__()
        print("Loading FastText model")

        self.word_embedding_model = fastText.load_model(FASTTEXT_EMBEDDIG_PATH)
        self.dimension_size = EMB_DIMENSION_SIZE

        self.tokenizer = get_tokenizer("basic_english")


    #There is no pre processing needed for Average Embedding
    def preprocess(self, list_of_tuples):
        pass

    #list_of_strings is an Iterable of tuples as strings
    def get_tuple_embedding(self, list_of_tuples):
        #This is an one liner for efficiency
        # returns a list of word embeddings for each token in a tuple
        #   self.word_embedding_model.get_word_vector(token) for token in self.tokenizer(tuple)
        # next we convert the list of word embeddings to a numpy array using np.array(list)
        # next we compute the element wise mean via np.mean(np.array([embeddings]), axis=0)
        # we repeat this process for all tuples in list_of_tuples
        #       for tuple in list_of_tuples
        # then convert everything to a numpy array at the end through np.array([ ])
        # So if you send N tuples, then this will return a numpy matrix of size N x D where D is embedding dimension
        average_embeddings = np.array([np.mean(np.array([self.word_embedding_model.get_word_vector(token) for token in self.tokenizer(_tuple)]), axis=0) for _tuple in list_of_tuples])
        return average_embeddings

    #Return word embeddings for a list of words
    def get_word_embedding(self, list_of_words):
        return [self.word_embedding_model.get_word_vector(word) for word in list_of_words]

#Fast Text+SIF DeepBlocking 源码
class SIFEmbedding(ABCTupleEmbedding):
    # sif_weighting_param is a parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    # the SIF paper set the default value to 1e-3
    # remove_pc is a Boolean parameter that controls whether to remove the first principal component or not
    # min_freq: if a word is too infrequent (ie frequency < min_freq), set a SIF weight of 1.0 else apply the formula
    #   The SIF paper applies this formula if the word is not the top-N most frequent
    def __init__(self, sif_weighting_param=1e-3, remove_pc=True, min_freq=0):
        super().__init__()
        print("Loading FastText model")

        self.word_embedding_model = fastText.load_model(FASTTEXT_EMBEDDIG_PATH)
        self.dimension_size = EMB_DIMENSION_SIZE

        self.tokenizer = get_tokenizer("basic_english")

        # Word to frequency counter
        self.word_to_frequencies = Counter()

        # Total number of distinct tokens
        self.total_tokens = 0

        self.sif_weighting_param = sif_weighting_param
        self.remove_pc = remove_pc
        self.min_freq = min_freq

        self.token_weight_dict = {}

    # There is no pre processing needed for Average Embedding
    def preprocess(self, list_of_tuples):
        for tuple_as_str in list_of_tuples:
            self.word_to_frequencies.update(self.tokenizer(tuple_as_str))

        # Count all the tokens in each tuples
        self.total_tokens = sum(self.word_to_frequencies.values())

        # Compute the weight for each token using the SIF scheme
        a = self.sif_weighting_param
        for word, frequency in self.word_to_frequencies.items():
            if frequency >= self.min_freq:
                self.token_weight_dict[word] = a / (a + frequency / self.total_tokens)
            else:
                self.token_weight_dict[word] = 1.0

    # list_of_strings is an Iterable of tuples as strings
    # See the comments of AverageEmbedding's get_tuple_embedding for details about how this works
    def get_tuple_embedding(self, list_of_tuples):
        num_tuples = len(list_of_tuples)
        tuple_embeddings = np.zeros((num_tuples, self.dimension_size))

        for index, _tuple in enumerate(list_of_tuples):
            # Compute a weighted average using token_weight_dict
            tuple_embeddings[index, :] = np.mean(np.array(
                [self.word_embedding_model.get_word_vector(token) * self.token_weight_dict[token] for token in
                 self.tokenizer(_tuple)]), axis=0)

        # From the code of the SIF paper at
        # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        if self.remove_pc:
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(tuple_embeddings)
            pc = svd.components_

            sif_embeddings = tuple_embeddings - tuple_embeddings.dot(pc.transpose()) * pc
        else:
            sif_embeddings = tuple_embeddings
        return sif_embeddings

    def get_word_embedding(self, list_of_words):
        return [self.word_embedding_model.get_word_vector(word) for word in list_of_words]


#AE deepBlocking的源码
class DB_AutoEncoderTupleEmbedding(ABCTupleEmbedding):
    def __init__(self, hidden_dimensions=(2 * AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE)):
        super().__init__()
        self.input_dimension = EMB_DIMENSION_SIZE
        self.hidden_dimensions = hidden_dimensions
        self.sif_embedding_model = SIFEmbedding()

    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        print("Training AutoEncoder model")
        self.sif_embedding_model.preprocess(list_of_tuples)
        embedding_matrix = self.sif_embedding_model.get_tuple_embedding(list_of_tuples)
        trainer = dl_models.DB_AutoEncoderTrainer(self.input_dimension, self.hidden_dimensions)
        self.autoencoder_model = trainer.train(embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(self.sif_embedding_model.get_tuple_embedding(list_of_tuples)).float()
        return self.autoencoder_model.get_tuple_embedding(embedding_matrix)

    # This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        embedding_matrix = torch.tensor(self.sif_embedding_model.get_tuple_embedding(list_of_words)).float()
        return self.autoencoder_model.get_tuple_embedding(embedding_matrix)

# CTT deeoBlocking的源码
class CTTTupleEmbedding(ABCTupleEmbedding):
    def __init__(self, hidden_dimensions=(2 * AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE),
                 synth_tuples_per_tuple=5, pos_to_neg_ratio=1, max_perturbation=0.4):
        super().__init__()
        self.input_dimension = EMB_DIMENSION_SIZE
        self.hidden_dimensions = hidden_dimensions
        self.synth_tuples_per_tuple = synth_tuples_per_tuple
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.max_perturbation = max_perturbation

        # By default, CTT uses SIF as the aggregator
        self.sif_embedding_model = SIFEmbedding()

    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        print("Training CTT model")
        a = list(list_of_tuples)
        self.sif_embedding_model.preprocess(list_of_tuples)

        left_tuple_list, right_tuple_list, label_list = generate_synthetic_training_data(list_of_tuples,
                                                                                         self.synth_tuples_per_tuple,
                                                                                         self.pos_to_neg_ratio,
                                                                                         self.max_perturbation)

        self.left_embedding_matrix = self.sif_embedding_model.get_tuple_embedding(left_tuple_list)
        self.right_embedding_matrix = self.sif_embedding_model.get_tuple_embedding(right_tuple_list)
        self.label_list = label_list

        trainer = dl_models.CTTModelTrainer(self.input_dimension, self.hidden_dimensions)
        self.ctt_model = trainer.train(self.left_embedding_matrix, self.right_embedding_matrix, self.label_list,
                                       num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(self.sif_embedding_model.get_tuple_embedding(list_of_tuples)).float()
        return embedding_matrix

    # This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        embedding_matrix = torch.tensor(self.sif_embedding_model.get_tuple_embedding(list_of_words)).float()
        return embedding_matrix


class SentenceBertTupleEmbedding(ABCTupleEmbedding):
    def __init__(self):
        super().__init__()
        self.sentence_bert_model = SentenceTransformer('all-mpnet-base-v2',device=get_device()).eval()


    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        print("load pre-trained SentenceBert model")
        self.sentence_bert_model = self.sentence_bert_model.eval()

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        #不更新参数
        with torch.no_grad():
            #Sentences are encoded by calling model.encode()
            sentence_embeddings = self.sentence_bert_model.encode(list_of_tuples,device=get_device())
            return sentence_embeddings


class AutoEncoderTupleEmbedding(ABCTupleEmbedding):
    def __init__(self, hidden_dim):
        super().__init__()
        self.input_dimension = 768
        self.hidden_dim = hidden_dim
        self.sentence_bert_model = SentenceBertTupleEmbedding()
    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        print("Training AutoEncoder model")
        embedding_matrix = self.sentence_bert_model.get_tuple_embedding(list_of_tuples)
        ######
        trainer = dl_models.AutoEncoderTrainer(self.input_dimension, self.hidden_dim)
        self.autoencoder_model = trainer.train(embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(self.sentence_bert_model.get_tuple_embedding(list_of_tuples)).float()
        embedding_matrix = embedding_matrix.cuda()
        return self.autoencoder_model.get_tuple_embedding(embedding_matrix)


class AutoEncoder_V2TupleEmbedding(ABCTupleEmbedding):
    def __init__(self, hidden_dim):
        super().__init__()
        self.input_dimension = 768
        self.hidden_dim = hidden_dim
        self.sentence_bert_model = SentenceBertTupleEmbedding()
    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        print("Training AutoEncoder model")
        embedding_matrix = self.sentence_bert_model.get_tuple_embedding(list_of_tuples)
        ######
        trainer = dl_models.AutoEncoder_V2Trainer(self.input_dimension, self.hidden_dim)
        self.autoencoder_model = trainer.train(embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(self.sentence_bert_model.get_tuple_embedding(list_of_tuples)).float()
        embedding_matrix = embedding_matrix.cuda()
        return self.autoencoder_model.get_tuple_embedding(embedding_matrix)


#使用不同层SBert得到embedding
class SentenceBert_layer(ABCTupleEmbedding):
    def __init__(self,layer, model_name):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import Counter
        from nltk.corpus import stopwords

        stopwords = set(stopwords.words('english'))
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.layer = layer
        fn = "/home/cyp/uB-CL-main/uB-CL-main/data/data1/all_data_merge.csv"  # Assuming the dataset is in the "data" folder
        # 这个all_data_merge是原始记录加会议数据对吧 ok
            # dataset is the filename
        content = []
        with open(fn) as fin:
            for line in fin:
                LL = line.split('\t')
                for entry in LL:
                    content.append(entry.lower())
            # Create TF-IDF vectorizer
        # vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[\w\.\-"]+\b').fit(content)
        vectorizer = TfidfVectorizer().fit(content)
        self.vocab = vectorizer.vocabulary_
        self.idf = vectorizer.idf_

    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        # 用不到
        print("load pre-trained SentenceBert model")
        self.sentence_bert_model = self.model.eval()
    # def summarize_text(self, text, topk=126, max_bert_len=126):
    #     """
    #     对给定文本进行摘要操作，保留 tf-idf 权重最高的 token。
    #     参数:
    #         topk: 用于候选 token 选择的数量阈值。
    #         max_bert_len: 摘要允许的最大 BERT 分词长度（即所有候选 token 的分词长度总和）。
    #     """
    #     tokens = text.split()
    #     if self.get_len(text) <= max_bert_len:
    #         return text
    #     cnt = Counter()

    #     # 计算每个 token 的 tf-idf 权重（仅考虑非停用词且在词汇表中的 token）token not in stopwords and 
    #     for token in tokens:
    #         if token in self.vocab:
    #             cnt[token] = self.idf[self.vocab[token]]
                        
    #     # 获取 tf-idf 权重最高的前 topk 个 token，形成候选 token 集合
    #     topk_tokens = {token for token, _ in cnt.most_common(topk)}

    #     # 根据 BERT 分词长度控制摘要的总长度
    #     summary_tokens = []
    #     current_length = 0
    #     for token in tokens:
    #         # 如果 token 在候选集合中，则考虑加入摘要
    #         if token in topk_tokens:
    #             token_len = self.get_len(token)  # 获取 token 的 BERT 分词长度
    #             if current_length + token_len <= max_bert_len:
    #                 summary_tokens.append(token)
    #                 current_length += token_len
    #             # 如果累计长度达到或超过 max_bert_len，则停止添加
    #             if current_length >= max_bert_len:
    #                 break

    #     return ' '.join(summary_tokens).strip()
    def summarize_text(self, text, topk=126, max_bert_len=126):
        """
        对给定文本进行摘要操作，在保留原始顺序的前提下优先保留高IDF值的token。
        参数:
            topk: 用于候选token选择的数量阈值。
            max_bert_len: 摘要允许的最大BERT分词长度。
        """
        tokens = text.split()
        if self.get_len(text) <= max_bert_len:
            return text

        # 预处理：过滤停用词和不在词汇表的token，并记录IDF和长度
        processed = []
        for token in tokens:
            if token in self.vocab:
                token_idf = self.idf[self.vocab[token]]
                token_len = self.get_len(token)
                processed.append((token, token_idf, token_len, len(processed)))  # 保留原始索引
            else:
                processed.append((token, -1, self.get_len(token), len(processed)))    # 标记无效token

        # 筛选候选token (IDF最高的topk个，按原始顺序排列)
        valid_candidates = [item for item in processed if item[1] != -1]
        valid_candidates.sort(key=lambda x: (-x[1], x[3]))  # 按IDF降序，原始索引升序排序
        topk_candidates = valid_candidates[:topk]
        topk_candidates.sort(key=lambda x: x[3])  # 恢复原始顺序

        # 动态选择token并控制长度
        selected = []
        current_length = 0
        for token_info in topk_candidates:
            token, token_idf, token_len, _ = token_info
            
            if current_length + token_len <= max_bert_len:
                selected.append(token_info)
                current_length += token_len
            else:
                # 计算需要腾出的空间
                needed_space = current_length + token_len - max_bert_len
                
                # 收集可删除的候选(按IDF升序，位置升序)
                sorted_candidates = sorted(
                    [(idx, t[1], t[2]) for idx, t in enumerate(selected)],
                    key=lambda x: (x[1], x[0])
                )
                
                # 尝试删除低价值token
                removed_length = 0
                removed_indices = []
                for candidate in sorted_candidates:
                    if removed_length >= needed_space:
                        break
                    
                    if (current_length - candidate[2] + token_len) <= max_bert_len:
                        removed_length += candidate[2]
                        removed_indices.append(candidate[0])
                
                # 执行删除操作
                if removed_length > 0:
                    for idx in sorted(removed_indices, reverse=True):
                        current_length -= selected[idx][2]
                        del selected[idx]
                    
                    # 添加当前token
                    if current_length + token_len <= max_bert_len:
                        selected.append(token_info)
                        current_length += token_len

        # 按原始顺序提取结果
        selected.sort(key=lambda x: x[3])  # 按原始索引排序
        summary_tokens = [item[0] for item in selected]
        
        return ' '.join(summary_tokens).strip()
    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_len(self, word):
        """Return the sentence_piece length of a token.
        """
        length = len(self.tokenizer.tokenize(word))
        return length
    def get_tuple_embedding(self, list_of_tuples):
        #不更新参数
        batch_size = 32
        all_embeddings=[]
        device = get_device()
        device = torch.device(device)
        self.model.to(device)
        for idx in tqdm(range(0,len(list_of_tuples),batch_size)):
            sentence_batch = list_of_tuples[idx:idx+batch_size]
            # 在这加摘要
            # sentence_batch = [self.summarize_text(each, 380, 380) for each in sentence_batch]
            sentence_batch = self.tokenizer(sentence_batch, padding="max_length",max_length=384,truncation=True,return_tensors='pt')
            features = batch_to_device(sentence_batch, device)
            with torch.no_grad():
                model_output = self.model(**features, output_hidden_states=True)
                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, sentence_batch['attention_mask'],self.layer)
                sentence_embeddings = sentence_embeddings.detach()
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                sentence_embeddings = sentence_embeddings.cpu()
                all_embeddings.extend(sentence_embeddings)
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        print(all_embeddings.shape)
        return all_embeddings

class SentenceBert_layer_dense_sparse(ABCTupleEmbedding):
    def __init__(self,layer, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.layer = layer
        self.sparse_linear = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.load_sparse_linear(model_name)

    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        # 用不到
        print("load pre-trained SentenceBert model")
        self.sentence_bert_model = self.model.eval()

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        #不更新参数
        batch_size = 32
        all_embeddings = []
        all_sparse_embeddings = []
        device = get_device()
        device = torch.device(device)
        self.model.to(device)
        self.sparse_linear.to(device)
        for idx in tqdm(range(0,len(list_of_tuples),batch_size)):
            sentence_batch = list_of_tuples[idx:idx+batch_size]
            sentence_batch = self.tokenizer(sentence_batch, padding="max_length", max_length=384, truncation=True, return_tensors='pt')
            features = batch_to_device(sentence_batch, device)
            with torch.no_grad():
                model_output = self.model(**features, output_hidden_states=True)
                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, sentence_batch['attention_mask'],self.layer)
                sentence_embeddings = sentence_embeddings.detach()
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                sentence_embeddings = sentence_embeddings.cpu()
                sentence_sparse_embeddings = self.sparse_embedding(model_output.last_hidden_state, sentence_batch['input_ids'])
                sentence_sparse_embeddings = sentence_sparse_embeddings.cpu()
                all_sparse_embeddings.extend(sentence_sparse_embeddings)
                all_embeddings.extend(sentence_embeddings)
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        # 转为稀疏向量节省空间
        all_sparse_embeddings = np.asarray([emb.numpy() for emb in all_sparse_embeddings])
        all_sparse_embeddings = csr_matrix(all_sparse_embeddings)

        print(all_embeddings.shape)
        print(all_sparse_embeddings.shape)
        return all_embeddings, all_sparse_embeddings

    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding: return token_weights

        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.model.config.vocab_size,
                                       dtype=token_weights.dtype,
                                       device=token_weights.device)
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding

    def load_sparse_linear(self, model_dir):
        sparse_state_dict = torch.load(os.path.join(model_dir, 'sparse_linear.pt'))
        self.sparse_linear.load_state_dict(sparse_state_dict)


class BART_layer(ABCTupleEmbedding):
    def __init__(self,layer, model_name, encoder_or_decoder="encoder"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BartModel.from_pretrained(model_name).eval()
        self.layer = layer
        self.encoder_or_decoder = encoder_or_decoder
        print(f"采用bart {encoder_or_decoder}")

    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        # 用不到
        print("load pre-trained SentenceBert model")
        self.sentence_bert_model = self.model.eval()

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        #不更新参数
        batch_size = 32
        all_embeddings=[]
        device = get_device()
        device = torch.device(device)
        self.model.to(device)
        for idx in tqdm(range(0,len(list_of_tuples),batch_size)):
            sentence_batch = list_of_tuples[idx:idx+batch_size]
            sentence_batch = self.tokenizer(sentence_batch, padding="max_length",max_length=384,truncation=True,return_tensors='pt')
            features = batch_to_device(sentence_batch, device)
            with torch.no_grad():
                model_output = self.model(**features,output_hidden_states=True)
                # Perform pooling
                sentence_embeddings = bart_mean_pooling(model_output, sentence_batch['attention_mask'], self.layer, self.encoder_or_decoder)
                sentence_embeddings = sentence_embeddings.detach()
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                sentence_embeddings = sentence_embeddings.cpu()
                all_embeddings.extend(sentence_embeddings)
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        print(all_embeddings.shape)
        return all_embeddings

#使用不同层的bert得到embeddings
class Bert_layer(ABCTupleEmbedding):
    def __init__(self,layer, model_name):
        super().__init__()
        print("load pre-trained Bert model")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).eval()
        self.layer = layer

    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        #用不到
        print("load pre-trained Bert model")
        self.sentence_bert_model = self.model.eval()

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        #不更新参数
        batch_size = 64
        all_embeddings=[]
        device = get_device()
        device = torch.device(device)
        self.model.to(device)
        for idx in tqdm(range(0,len(list_of_tuples),batch_size)):
            sentence_batch = list_of_tuples[idx:idx+batch_size]
            sentence_batch = self.tokenizer(sentence_batch, padding="max_length",max_length=384,truncation=True,return_tensors='pt')
            features = batch_to_device(sentence_batch, device)
            with torch.no_grad():
                model_output = self.model(**features,output_hidden_states=True)
                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, sentence_batch['attention_mask'],self.layer)  # 对每个token的hidden_state进行mean pooling
                sentence_embeddings = sentence_embeddings.detach()
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)   # 对每个句子的embedding进行L2归一化
                sentence_embeddings = sentence_embeddings.cpu()
                all_embeddings.extend(sentence_embeddings)
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def get_tuple_embedding_cls(self, list_of_tuples):
        #不更新参数
        batch_size = 32
        all_embeddings=[]
        device = get_device()
        device = torch.device(device)
        self.model.to(device)
        for idx in tqdm(range(0,len(list_of_tuples),batch_size)):
            sentence_batch = list_of_tuples[idx:idx+batch_size]
            sentence_batch = self.tokenizer(sentence_batch, padding="max_length",max_length=384,truncation=True,return_tensors='pt')
            features = batch_to_device(sentence_batch, device)
            with torch.no_grad():
                model_output = self.model(**features,output_hidden_states=True)
                # 只提取指定layer的cls token的hidden_state
                sentence_embeddings = model_output.hidden_states[self.layer][:,0,:]  # 提取指定层的cls token的hidden_state
                sentence_embeddings = sentence_embeddings.detach()
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)   # 对每个句子的embedding进行L2归一化
                sentence_embeddings = sentence_embeddings.cpu()
                all_embeddings.extend(sentence_embeddings)
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings



#使用VAE得到embedding
class VAETupleEmbedding(ABCTupleEmbedding):
    def __init__(self, hidden_dim):
        super().__init__()
        self.input_dimension = 768
        self.hidden_dim = hidden_dim
        self.sentence_bert_model = SentenceBertTupleEmbedding()
    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        print("Training AutoEncoder model")
        embedding_matrix = self.sentence_bert_model.get_tuple_embedding(list_of_tuples)
        ######
        trainer = dl_models.VAETrainer(self.input_dimension, self.hidden_dim)
        self.VAE_model = trainer.train(embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(self.sentence_bert_model.get_tuple_embedding(list_of_tuples)).float()
        embedding_matrix = embedding_matrix.cuda()
        return self.VAE_model.get_tuple_embedding(embedding_matrix)



class Doc2Vec(ABCTupleEmbedding):
    def __init__(self,dataset,mode="pre"):
        super().__init__()
        model_path = Doc2Vec_model_path
        self.dataset = dataset
        self.start_alpha = start_alpha
        self.infer_epoch = infer_epoch
        self.doc2vec = g.Doc2Vec.load(model_path)
        self.dataset=dataset
        self.mode = mode


    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    # fine_train=100,train=500
    def preprocess(self, list_of_tuples):
        print("generate txt")
        self.fileName = "record.txt"
        with open(self.fileName, 'w', encoding='utf-8') as file:
            for i in list_of_tuples:
                i = i.lower()
                i = i+"\n"
                file.write(i)
        if self.mode == "fine_train":
            vector_size = 300
            window_size = 15
            min_count = 1
            sampling_threshold = 1e-5
            negative_size = 5
            train_epoch = 100
            dm = 0  # 0 = dbow; 1 = dmpv
            worker_count = 1  # number of parallel processes
            pre_path = os.path.join("dic", self.dataset + "_pre.txt")
            # train doc2vec model
            docs = g.doc2vec.TaggedLineDocument(self.fileName)
            model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count,
                              sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size,
                              dbow_words=1, dm_concat=1, pretrained_emb=pre_path, iter=train_epoch)
            self.doc2vec = model
        elif self.mode == "train":
            vector_size = 300
            window_size = 15
            min_count = 1
            sampling_threshold = 1e-5
            negative_size = 5
            train_epoch = 500
            dm = 0  # 0 = dbow; 1 = dmpv
            worker_count = 1  # number of parallel processes
            pre_path = os.path.join("dic", self.dataset + "_pre.txt")
            # train doc2vec model
            docs = g.doc2vec.TaggedLineDocument(self.fileName)
            model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count,
                              sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size,
                              dbow_words=1, dm_concat=1, pretrained_emb=None, iter=train_epoch)
            self.doc2vec = model
    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        if self.dataset=="X2":
            self.fileName="X2record.txt"
        all_embeddings = []
        test_doc = [x.strip().split() for x in codecs.open(self.fileName,"r","utf-8").readlines()]
        for d in test_doc:
            all_embeddings.append(self.doc2vec.infer_vector(d,alpha=self.start_alpha,epochs=self.infer_epoch))
        all_embeddings = np.asarray(all_embeddings)

        return all_embeddings






#BERT+SIF
#Fast Text+SIF DeepBlocking 源码
class BERT_SIF_Embedding(ABCTupleEmbedding):
    # sif_weighting_param is a parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    # the SIF paper set the default value to 1e-3
    # remove_pc is a Boolean parameter that controls whether to remove the first principal component or not
    # min_freq: if a word is too infrequent (ie frequency < min_freq), set a SIF weight of 1.0 else apply the formula
    #   The SIF paper applies this formula if the word is not the top-N most frequent
    def __init__(self,layer, model_name, sif_weighting_param=1e-3, remove_pc=True, min_freq=0):
        super().__init__()
        print("Loading BERT model")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer
        self.word_embedding_model = BertModel.from_pretrained(model_name).eval().to(self.device)
        self.dimension_size = 768

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Word to frequency counter
        self.word_to_frequencies = Counter()

        # Total number of distinct tokens
        self.total_tokens = 0

        self.sif_weighting_param = sif_weighting_param
        self.remove_pc = remove_pc
        self.min_freq = min_freq

        self.token_weight_dict = {}

    # There is no pre processing needed for Average Embedding
    def preprocess(self, list_of_tuples):

        for tuple_as_str in list_of_tuples:
            self.word_to_frequencies.update(self.tokenizer(tuple_as_str)["input_ids"])

        # Count all the tokens in each tuples
        self.total_tokens = sum(self.word_to_frequencies.values())

        # Compute the weight for each token using the SIF scheme
        a = self.sif_weighting_param
        for word, frequency in self.word_to_frequencies.items():
            if frequency >= self.min_freq:
                self.token_weight_dict[word] = a / (a + frequency / self.total_tokens)
            else:
                self.token_weight_dict[word] = 1.0

    # list_of_strings is an Iterable of tuples as strings
    # See the comments of AverageEmbedding's get_tuple_embedding for details about how this works
    # def get_tuple_embedding(self, list_of_tuples):
    #     num_tuples = len(list_of_tuples)
    #     self.token_weight_dict[0] = 0.0
    #     tuple_embeddings = np.zeros((num_tuples, self.dimension_size))
    #
    #
    #     for index, _tuple in enumerate(list_of_tuples):
    #         #拿到句子中每个token的权重:
    #         BERT_input = self.tokenizer(_tuple,return_tensors="pt",padding="max_length",max_length=384,truncation=True)
    #         token_list = BERT_input["input_ids"][0].tolist()
    #         token_weight = np.array([self.token_weight_dict[x] for x in token_list])
    #         seq_len = torch.count_nonzero(BERT_input["attention_mask"][0],dim=0)
    #         #token_weight目前的维度为：1，max_length
    #         attention_mask =  BERT_input["attention_mask"][0].detach().numpy()
    #         token_weight = token_weight * attention_mask
    #         #将token_weight的维度转换为：max_length，1
    #         token_weight = token_weight.reshape(-1,1)
    #         #拿到句子中每个token的隐藏表示
    #         if BERT_input["input_ids"].shape[-1]>=512:
    #             print(BERT_input["input_ids"].shape)
    #         tokens_hidden_state = self.word_embedding_model(**BERT_input,output_hidden_states=True)["hidden_states"]
    #         tokens_hidden_state = tokens_hidden_state[self.layer][0].detach().numpy()
    #
    #         #Compute a weighted average using token_weight_dict
    #         tokens_hidden_state = tokens_hidden_state[0:seq_len]
    #         token_weight = token_weight[0:seq_len]
    #         tuple_embeddings[index, :] = np.mean(tokens_hidden_state*token_weight,axis=0)
    #
    #         # Compute a weighted average using token_weight_dict
    #         # tuple_embeddings[index, :] = np.mean(np.array(
    #         #     [self.word_embedding_model.get_word_vector(token) * self.token_weight_dict[token] for token in
    #         #      self.tokenizer(_tuple)]), axis=0)
    #
    #     # From the code of the SIF paper at
    #     # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
    #     if self.remove_pc:
    #         svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    #         svd.fit(tuple_embeddings)
    #         pc = svd.components_
    #
    #         sif_embeddings = tuple_embeddings - tuple_embeddings.dot(pc.transpose()) * pc
    #     else:
    #         sif_embeddings = tuple_embeddings
    #     return sif_embeddings
    def get_tuple_embedding(self, list_of_tuples):
        num_tuples = len(list_of_tuples)
        self.token_weight_dict[0] = 0.0
        tuple_embeddings = torch.zeros((num_tuples, self.dimension_size), device=self.device)

        for index, _tuple in tqdm(enumerate(list_of_tuples), total=num_tuples):
            # 拿到句子中每个token的权重:
            BERT_input = self.tokenizer(_tuple, return_tensors="pt", padding="max_length", max_length=384,
                                        truncation=True)
            BERT_input = {key: value.to(self.device) for key, value in BERT_input.items()}
            token_list = BERT_input["input_ids"][0].tolist()
            token_weight = np.array([self.token_weight_dict[x] for x in token_list])
            seq_len = torch.count_nonzero(BERT_input["attention_mask"][0], dim=0)
            # token_weight目前的维度为：1，max_length
            attention_mask = BERT_input["attention_mask"][0].detach().cpu().numpy()
            token_weight = token_weight * attention_mask
            # 将token_weight的维度转换为：max_length，1
            token_weight = token_weight.reshape(-1, 1)
            # 拿到句子中每个token的隐藏表示
            if BERT_input["input_ids"].shape[-1] >= 512:
                print(BERT_input["input_ids"].shape)
            tokens_hidden_state = self.word_embedding_model(**BERT_input, output_hidden_states=True)["hidden_states"]
            tokens_hidden_state = tokens_hidden_state[self.layer][0].detach().cpu().numpy()

            # Compute a weighted average using token_weight_dict
            tokens_hidden_state = tokens_hidden_state[0:seq_len]
            token_weight = token_weight[0:seq_len]
            weighted_hidden_state = torch.tensor(tokens_hidden_state * token_weight, device=self.device)
            tuple_embeddings[index, :] = torch.mean(weighted_hidden_state, dim=0)

        # From the code of the SIF paper at
        # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        if self.remove_pc:
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(tuple_embeddings.cpu().numpy())
            pc = torch.tensor(svd.components_).to(self.device)

            sif_embeddings = tuple_embeddings - tuple_embeddings.mm(pc.transpose(0, 1)) * pc
        else:
            sif_embeddings = tuple_embeddings
        return sif_embeddings.cpu()

    def get_word_embedding(self, list_of_words):
        return [self.word_embedding_model.get_word_vector(word) for word in list_of_words]



#Sentence_BERT+SIF
#Fast Text+SIF DeepBlocking 源码
class Sentence_BERT_SIF_Embedding(ABCTupleEmbedding):
    # sif_weighting_param is a parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    # the SIF paper set the default value to 1e-3
    # remove_pc is a Boolean parameter that controls whether to remove the first principal component or not
    # min_freq: if a word is too infrequent (ie frequency < min_freq), set a SIF weight of 1.0 else apply the formula
    #   The SIF paper applies this formula if the word is not the top-N most frequent
    def __init__(self,layer, sif_weighting_param=1e-3, remove_pc=True, min_freq=0):
        super().__init__()
        print("Loading Sentence-BERT model")
        self.layer = layer
        self.word_embedding_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').eval()
        self.dimension_size = 768

        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

        # Word to frequency counter
        self.word_to_frequencies = Counter()

        # Total number of distinct tokens
        self.total_tokens = 0

        self.sif_weighting_param = sif_weighting_param
        self.remove_pc = remove_pc
        self.min_freq = min_freq

        self.token_weight_dict = {}

    # There is no pre processing needed for Average Embedding
    def preprocess(self, list_of_tuples):

        for tuple_as_str in list_of_tuples:
            self.word_to_frequencies.update(self.tokenizer(tuple_as_str)["input_ids"])

        # Count all the tokens in each tuples
        self.total_tokens = sum(self.word_to_frequencies.values())

        # Compute the weight for each token using the SIF scheme
        a = self.sif_weighting_param
        for word, frequency in self.word_to_frequencies.items():
            if frequency >= self.min_freq:
                self.token_weight_dict[word] = a / (a + frequency / self.total_tokens)
            else:
                self.token_weight_dict[word] = 1.0

    # list_of_strings is an Iterable of tuples as strings
    # See the comments of AverageEmbedding's get_tuple_embedding for details about how this works
    def get_tuple_embedding(self, list_of_tuples):
        num_tuples = len(list_of_tuples)
        self.token_weight_dict[1] = 0.0
        tuple_embeddings = np.zeros((num_tuples, self.dimension_size))


        for index, _tuple in enumerate(list_of_tuples):
            #拿到句子中每个token的权重:
            BERT_input = self.tokenizer(_tuple,return_tensors="pt",padding="max_length",max_length=384,truncation=True)
            token_list = BERT_input["input_ids"][0].tolist()
            token_weight = np.array([self.token_weight_dict[x] for x in token_list])
            seq_len = torch.count_nonzero(BERT_input["attention_mask"][0],dim=0)
            #token_weight目前的维度为：1，max_length
            attention_mask =  BERT_input["attention_mask"][0].detach().numpy()
            token_weight = token_weight * attention_mask
            #将token_weight的维度转换为：max_length，1
            token_weight = token_weight.reshape(-1,1)
            #拿到句子中每个token的隐藏表示
            if BERT_input["input_ids"].shape[-1]>=512:
                print(BERT_input["input_ids"].shape)
            tokens_hidden_state = self.word_embedding_model(**BERT_input,output_hidden_states=True)["hidden_states"]
            tokens_hidden_state = tokens_hidden_state[self.layer][0].detach().numpy()

            #Compute a weighted average using token_weight_dict
            tokens_hidden_state = tokens_hidden_state[0:seq_len]
            token_weight = token_weight[0:seq_len]
            tuple_embeddings[index, :] = np.mean(tokens_hidden_state*token_weight,axis=0)

            # Compute a weighted average using token_weight_dict
            # tuple_embeddings[index, :] = np.mean(np.array(
            #     [self.word_embedding_model.get_word_vector(token) * self.token_weight_dict[token] for token in
            #      self.tokenizer(_tuple)]), axis=0)

        # From the code of the SIF paper at
        # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        if self.remove_pc:
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(tuple_embeddings)
            pc = svd.components_

            sif_embeddings = tuple_embeddings - tuple_embeddings.dot(pc.transpose()) * pc
        else:
            sif_embeddings = tuple_embeddings
        return sif_embeddings

    def get_word_embedding(self, list_of_words):
        return [self.word_embedding_model.get_word_vector(word) for word in list_of_words]