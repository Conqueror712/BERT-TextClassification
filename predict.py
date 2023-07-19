import time

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import argparse


class Config:
    # 配置参数

    def __init__(self, dataset):
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        # cru = os.path.dirname(__file__)
        # self.class_list = [str(i) for i in range(len(key))]  # 类别名单
        self.save_path = 'THUCNews/saved_dict/bert.ckpt'
        self.device = torch.device('cpu')
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

    def build_dataset(self, text):
        lin = text.strip()
        pad_size = len(lin)
        token = self.tokenizer.tokenize(lin)
        token = ['[CLS]'] + token
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask])


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


# config = Config()
# model = Model(config).to(config.device)
# model.load_state_dict(torch.load(config.save_path, map_location='cpu'))


def prediction_model(text):
    """输入一句问话预测"""
    data = config.build_dataset(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
    return key[int(num)]


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='bert', help='choose a model: Bert, ERNIE')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # 识别的类型
    key = {0: 'finance',
           1: 'realty',
           2: 'stocks',
           3: 'education',
           4: 'science',
           5: 'society',
           6: 'politics',
           7: 'sports',
           8: 'game',
           9: 'entertainment'
           }
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    config = Config(dataset)
    model = Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location='cpu'))
    print(">>>>>> Loading data... >>>>>>")
    test_text = '名师详解考研复试英语听力备考策略'
    print(test_text, " \t---分类预测结果--> ", prediction_model(test_text))
    test_text = '今年乃至明年上半年不会有新政策出台'
    print(test_text, " \t---分类预测结果--> ", prediction_model(test_text))
    test_text = '女子偷取男友28万元存款要挟结婚获刑'
    print(test_text, " \t---分类预测结果--> ", prediction_model(test_text))
