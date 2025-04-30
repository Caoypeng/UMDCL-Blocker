from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载数据，构建 TF-IDF 词典
fn = "/home/cyp/uB-CL-main/uB-CL-main/data/data1/all_data_merge.csv"
content = []
with open(fn, encoding="utf-8") as fin:
    for line in fin:
        LL = line.split('\t')
        for entry in LL:
            content.append(entry)

vectorizer = TfidfVectorizer(
    token_pattern=r'(?u)\S+',  # 匹配包含括号和连字符的标记
    lowercase=False,                 # 保留原始大小写
    stop_words=None                # 不使用停用词过滤
).fit(content)
vocab = vectorizer.vocabulary_
idf = vectorizer.idf_


class Summarizer:
    def __init__(self, vocab, idf):
        self.vocab = vocab
        self.idf = idf

    def get_len(self, text):
        """
        简单定义每个 token 的长度为1，文本长度为 token 数量
        """
        return len(text.split())

    def summarize_text(self, text, topk=126, max_bert_len=126):
        """
        对给定文本进行摘要操作，保留加权后的关键 token。
        新增规则：
            1. 位置权重：越靠前的 token 权重越高（这里采用前50%的 token 权重显著提升）
            2. 数字权重：如果 token 是数字，则权重翻倍
        """
        tokens = text.split()
        # 如果文本 token 数量未超过 max_bert_len，则直接返回原文

        
        cnt = Counter()
        total_tokens = len(tokens)

        # 计算每个 token 的 tf-idf 权重（仅考虑非停用词且在词汇表中的 token）
        for idx, token in enumerate(tokens):
            if token in self.vocab :
                print(token)
                score = self.idf[self.vocab[token]]


                # 规则1：位置权重——越靠前的 token 权重越高
                # 这里采用前50%的 token 乘以 10.0
                if idx < total_tokens * 0.5:
                    score *= 10.0

                # 规则2：数字权重——如果 token 是数字，则进一步提高权重
                if token.isdigit():
                    score *= 2.0
                cnt[token] += score

        # for token, score in cnt.items():
        #     print(f"Token: {token}, Score: {score}")
        # 获取 tf-idf（融合了额外规则）权重最高的前 topk 个 token 作为候选 token 集合
        topk_tokens = {token for token, _ in cnt.most_common(topk)}
        print(topk_tokens)
        # 根据 BERT 分词长度控制摘要的总长度（这里简单假设每个 token 长度为1）
        summary_tokens = []
        current_length = 0
        for token in tokens:
            if token in topk_tokens:
                token_len = 1  # 这里简化为每个 token 长度为 1
                if current_length + token_len <= max_bert_len:
                    summary_tokens.append(token)
                    current_length += token_len
                if current_length >= max_bert_len:
                    break

        return ' '.join(summary_tokens).strip()

# 测试代码
if __name__ == "__main__":
    summarizer = Summarizer(vocab, idf)
    test_text = ("HP DSS Software - (v. 4.0) - Complete Package (t1936aa ua0) - Marketing Information: HP Digital Sending 4.0 software improves core business processes. ")
    summary = summarizer.summarize_text(test_text, topk=20, max_bert_len=20)
    print("Original Text:")
    print(test_text)
    print("\nSummary:")
    print(summary)
