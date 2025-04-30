import torch
import torch.nn as nn
import torch.distributed as dist
import os
from transformers.models.mpnet.modeling_mpnet import MPNetPreTrainedModel, MPNetModel, MPNetLMHead
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer
from transformers.file_utils import WEIGHTS_NAME, is_torch_tpu_available
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class CLModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "avg"]

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        return last_hidden

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)

        return x



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)  # 计算两个张量之间的余弦相似度

    def forward(self, x, y):
        return self.cos(x, y)
class AttentionWeight(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWeight, self).__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),  # 首先扩大维度
            nn.ReLU(),                              # 激活函数
            nn.Dropout(0.1),                        # Dropout以防止过拟合
            nn.Linear(hidden_dim * 2, hidden_dim)   # 再缩小回hidden_dim
        )
        
        self.key_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )


        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_sample, positive_matched_features):
        # Q, K 投影
        query = self.query_proj(query_sample)  # (batch_size, seq_len, hidden_dim)
        key = self.key_proj(query_sample)  # (batch_size, seq_len, hidden_dim)

        # 计算注意力分数 (Q * K^T / sqrt(d_k))
        d_k = query.size(-1)
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / (d_k ** 0.5)  # (batch_size, seq_len, seq_len)
        diag_attention = attention_scores.diagonal(dim1=-2, dim2=-1) 
        attention_weights = self.softmax(diag_attention/0.05)  # (batch_size, seq_len)

        return attention_weights


class SimilarityGateMechanism(nn.Module):
    def __init__(self, hidden_dim=8):
        super(SimilarityGateMechanism, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.glu = nn.GLU(dim=-1)
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sim1, sim2):
        combined_input = torch.cat((sim1, sim2), dim=-1)  # (batch_size, 2)
        hidden = self.fc1(combined_input)
        gate_weight=self.glu(hidden)
        gate_weight = self.fc2(gate_weight)
        gate_weight = self.sigmoid(gate_weight)
        return gate_weight 


class AdaptiveuBCLLoss(nn.Module):
    def __init__(self, initial_lambda=10.0, lambda_shape=(1,), hard_negative=False):
        super(AdaptiveuBCLLoss, self).__init__()
        self.sim = Similarity()
        initial_lambda_matrix = torch.full(lambda_shape, initial_lambda, dtype=torch.float32)
        self.lambda_ = nn.Parameter(initial_lambda_matrix)
        self.hard_negative = hard_negative
        self.gate_mechanism = SimilarityGateMechanism(2)  # 门控机制  
        self.weight=AttentionWeight(768)
    def forward(self, output):
        if self.hard_negative:
            z1, z2 = output[:, 0], output[:, 1]
            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            pos = cos_sim.diag().reshape(-1, 1)
            temp_res = self.lambda_ * (cos_sim - pos)
            loss = torch.log(torch.sum(torch.exp(temp_res), dim=1, keepdim=True))
            loss = torch.mean(loss)
        else:
            reshaped_tensor = output.view(32, 2, -1, 768)  
            toknes=reshaped_tensor.size(2)
            split_tensors = reshaped_tensor.split(1, dim=1)
            split_tensors = [x.squeeze(1) for x in split_tensors]
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         
            
            query_sample=split_tensors[0]   #  (32,32,768)
            
            query_sample_cls = split_tensors[0][:,0,:]  #  (32,768)
            query_sample_squrse=split_tensors[0][:,1:,:] #  (32,31,768)

            positive_sample=split_tensors[1].to(device)
            positive_sample_cls = split_tensors[1][:,0,:]
            positive_sample_sqursee=split_tensors[1][:,1:,:]
            batch_size = query_sample.size(0)


            neg_samples = []
            for i in range(batch_size):
            #获取当前查询 query_sample[i]，其余的查询作为负样本
                neg_sample = torch.cat([query_sample[:i], query_sample[i+1:]], dim=0)  # (batch_size-1, seq_len, hidden_dim)
                neg_samples.append(neg_sample)

            negative_samples = torch.stack(neg_samples, dim=0)
            negative_samples = negative_samples.split(1, dim=1)
            negative_samples = [x.squeeze(1) for x in negative_samples]

            query_expanded = query_sample_squrse.unsqueeze(2)   #  (32,31,1,768)

            positive_expanded = positive_sample_sqursee.unsqueeze(1)  #  (32,1,768)
            
            similarity_positive_sparse = self.sim(query_expanded, positive_expanded)  # (32,31,31)
            max_similarity_positive, max_indices  = similarity_positive_sparse.max(dim=2, keepdim=True)   # (32,31,1)
            positive_matched_features = torch.gather(
                positive_sample_sqursee, 
                dim=1,
                index=max_indices.expand(-1,-1,768)
            )

            attn_weights = self.weight(query_sample_squrse,positive_matched_features)  # (B,N)

            positive_similarity_sparse = max_similarity_positive.squeeze(2) # (32,31)

            similarity_positive_cls=self.sim(query_sample_cls, positive_sample_cls).unsqueeze(1)    #(32,1)

            positive_similarity=attn_weights*positive_similarity_sparse
            positive_similarity_sum = positive_similarity.sum(dim=1).unsqueeze(1)  #(32,1)
        
            gate_weights = self.gate_mechanism(similarity_positive_cls, positive_similarity_sum)

            positive_similarity_sum =  gate_weights * similarity_positive_cls + (1-gate_weights)* positive_similarity_sum

            negative_similarity_sums=[]

            for negative_sample in negative_samples:
                negative_sample_cls = negative_sample[:,0,:]
                negative_sample_squrse=negative_sample[:,1:,:]           

                negative_expanded = negative_sample_squrse.unsqueeze(1)

                similarity_negative = self.sim(query_expanded, negative_expanded)

                max_similarity_negative, max_indices = similarity_negative.max(dim=2, keepdim=True)
                negative_matched_features = torch.gather(
                    negative_sample_squrse, 
                    dim=1,
                    index=max_indices.expand(-1,-1,768)
                )
                negative_similarity_squrse = max_similarity_negative.squeeze(2)

                similarity_negative_cls=self.sim(query_sample_cls, negative_sample_cls).unsqueeze(1)
                attn_weights = self.weight(query_sample_squrse,negative_matched_features)  # (B,N)

                weighted_similarity_neg=attn_weights*negative_similarity_squrse

                negative_similarity_sum = weighted_similarity_neg.sum(dim=1).unsqueeze(1)
               
                gate_weights = self.gate_mechanism(similarity_negative_cls, negative_similarity_sum)
                similarity_final =gate_weights*similarity_negative_cls + (1-gate_weights)*negative_similarity_sum    

                negative_similarity_sums.append(similarity_final)
            negative_similarity_sum = torch.cat(negative_similarity_sums, dim=1) 


            pos_neg_sim_sum = torch.cat([positive_similarity_sum, negative_similarity_sum], dim=1)
            # 计算自适应温度参数
            alpha=1
            beta=0.1
            gamma=1
            sigma = torch.std(pos_neg_sim_sum, dim=1, unbiased=False)  # (batch_size,)
            tau = alpha * sigma + beta  # (batch_size,)
            tau = tau.unsqueeze(1)  # (batch_size, 1)
            
            # 数值稳定处理：减去最大值
            max_s = torch.max(pos_neg_sim_sum, dim=1, keepdim=True)[0]  # (batch_size, 1)
            s_p_norm = positive_similarity_sum - max_s
            s_n_norm = negative_similarity_sum - max_s

            # 计算动态权重
            delta = s_n_norm - s_p_norm  # (batch_size, num_neg)
            weights = F.softmax(gamma * delta, dim=1)  # 自动归一化
            
            # 计算指数项
            exp_sp = torch.exp(s_p_norm / tau)  # (batch_size, 1)
            exp_sn = torch.exp(s_n_norm / tau)  # (batch_size, num_neg)
            
            # 加权负样本项
            weighted_sn = (weights * exp_sn).sum(dim=1, keepdim=True)  # (batch_size, 1)
            
            # 计算最终损失
            numerator = exp_sp
            denominator = numerator + weighted_sn
            loss = -torch.log(numerator / denominator).mean()

        return loss



class MyMPNetModel(MPNetModel):
    def save_pretrained(self, save_directory, state_dict=None):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        if getattr(self.config, "xla_device", False) and is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            if xm.is_master_ordinal():
                # Save configuration file
                model_to_save.config.save_pretrained(save_directory)
            # xm.save takes care of saving only from master
            xm.save(state_dict, output_model_file)
        else:
            model_to_save.config.save_pretrained(save_directory)
            torch.save(state_dict, output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

class MPNetForCL(MPNetPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.model_type == "mpnet":
            self.mpnet = MyMPNetModel(config, add_pooling_layer=False)
        else:
            raise NotImplementedError

        self.pool_type = self.model_args.pool_type
        self.pooler = Pooler(self.model_args.pool_type)
        if self.model_args.pool_type == "cls":
            self.mlp = MLPLayer(config)
        self.init_weights()

        self.sparse_linear = nn.Linear(in_features=self.mpnet.config.hidden_size, out_features=1)
        if self.model_args.loss_type == "adaptive":
            self.loss_fct = AdaptiveuBCLLoss(self.model_args.initial_lambda, self.model_args.lambda_shape)
            if self.model_args.use_sparse:
                print("use sparse")
                self.loss_fct_sparse = AdaptiveuBCLLoss(self.model_args.initial_lambda, self.model_args.lambda_shape)
        elif self.model_args.loss_type == "simcse":
            self.loss_fct = SimCSELoss(self.model_args.temperature)
            if self.model_args.use_sparse:
                self.loss_fct_sparse = SimCSELoss(self.model_args.temperature)
        else:
            raise NotImplementedError
        self.vocab_size = self.mpnet.config.vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        if os.path.exists(os.path.join(self.model_args.model_name_or_path, 'sparse_linear.pt')):
            logger.info('loading existing sparse_linear---------')
            self.load_pooler(model_dir=self.model_args.model_name_or_path)
        else:
            logger.info(
                'The parameters of sparse linear is new initialize. Make sure the model is loaded for training, not inferencing')


    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding: return token_weights

        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size,
                                       dtype=token_weights.dtype,
                                       device=token_weights.device)
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding

    def encode(self, features):
        if features is None:
            return None
        batch_size = features["input_ids"].size(0)
        num_sent = features["input_ids"].size(1)
        features["input_ids"] = features["input_ids"].view((-1, features["input_ids"].size(-1)))
        features["attention_mask"] = features["attention_mask"].view((-1, features["attention_mask"].size(-1)))
        if "token_type_ids" in features:
            features["token_type_ids"] = features["token_type_ids"].view((-1, features["token_type_ids"].size(-1)))
        outputs = self.mpnet(**features)
        pooler_output = self.pooler(features["attention_mask"], outputs)

        if self.model_args.use_sparse:
            sparse_vecs = self.sparse_embedding(outputs.last_hidden_state, features['input_ids'])
            sparse_vecs = sparse_vecs.view((batch_size, num_sent, sparse_vecs.size(-1)))  # [batch_size, num_sent, vocab_size]


        else:
            sparse_vecs = None
        return pooler_output, sparse_vecs, outputs


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooler_output, sparse_vecs, outputs= self.encode(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "head_mask": head_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict
            }
        )
        loss = self.loss_fct(pooler_output)

        return CLModelOutput(
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save(self, output_dir: str):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            return state_dict

        self.mpnet.save_pretrained(output_dir, state_dict=_trans_state_dict(self.mpnet.state_dict()))
        if self.model_args.use_sparse:
            torch.save(_trans_state_dict(self.sparse_linear.state_dict()), os.path.join(output_dir, 'sparse_linear.pt'))
        torch.save(_trans_state_dict(self.loss_fct.weight.query_proj.state_dict()), os.path.join(output_dir, 'attention_query.pt'))  #
        torch.save(_trans_state_dict(self.loss_fct.weight.key_proj.state_dict()), os.path.join(output_dir, 'attention_key.pt'))



    def load_pooler(self, model_dir):
        sparse_state_dict = torch.load(os.path.join(model_dir, 'sparse_linear.pt'), map_location='cpu')
        self.sparse_linear.load_state_dict(sparse_state_dict)

