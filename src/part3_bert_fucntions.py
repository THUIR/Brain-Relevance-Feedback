import torch
from transformers import AutoTokenizer,BertForSequenceClassification
import json

def calc_score(tokenizer, model, query_dic, doc_dic, idx2word, cuda = torch.device('cuda:1')):
    query_list = ['']
    for k, v in query_dic.items():
        for i in range(v):
            query_list[0] += idx2word[k]
    doc_list = ['']
    for k, v in doc_dic.items():
        for i in range(v):
            doc_list[0] += idx2word[k]
    merged_list = []
    for i in range(len(query_list)):
        merged_list.append((query_list[i], doc_list[i]))
    encoded_input = tokenizer.batch_encode_plus(merged_list, truncation='only_second', max_length=512, padding=True, return_tensors="pt")
    encoded_input.to(cuda)
    with torch.no_grad():
        logits = model(**encoded_input).logits
    return [item[0] for item in logits]

def calc_score2(tokenizer, model, query, doc, cuda = torch.device('cuda:1')):
    query_list = [query]
    doc_list = [doc]
    merged_list = []
    for i in range(len(query_list)):
        merged_list.append((query_list[i], doc_list[i]))
    encoded_input = tokenizer.batch_encode_plus(merged_list, truncation='only_second', max_length=512, padding=True, return_tensors="pt")
    encoded_input.to(cuda)
    with torch.no_grad():
        logits = model(**encoded_input).logits
    return [item[0] for item in logits]

if __name__ == '__main__':
    # the checkpoint of pretrained BERT-based re-ranker can be downloaded from https://github.com/THUIR/T2Ranking
    ranker = BertForSequenceClassification.from_pretrained('../resource/swh-checkpoint-1500')
    tokenizer = AutoTokenizer.from_pretrained('../resource/chinese_bert_wwm', use_fast=True)
    device = torch.device('cuda:0')
    ranker.to(device)
    q = '你好北京'
    d1 = '北京欢迎你'
    d2 =  '可爱的小天使'
    print(calc_score2(tokenizer, ranker, q, d1, device))
    print(calc_score2(tokenizer, ranker, q, d2, device))


