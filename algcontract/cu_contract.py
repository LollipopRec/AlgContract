from email.errors import BoundaryError
import json
import jieba
import numpy as np
import tensorflow.keras as K
import tensorflow as tf
import os
import math
import random
import heapq
from algorithm_contract import ContentUnderstandingAlgorithmContract

def random_list(k):
    output = []
    while len(output) < k:
        num = random.randint(0, 99)
        if num not in output:
            output.append(num)
    return output 

def get_ar_file(ar_ads):
    ar_local_ads = ar_ads
    return ar_local_ads

def list_dict_json_file(inputs, path):
    with open(path, "w") as f:
        for key in inputs:
            line = json.dumps(key, ensure_ascii=False)
            f.write(line + "\n")


class DataReader(object):
    # 将原始数据处理成模型可以接受的数据格式
    def __init__(self, data_path, vocab_path, text_len, label_len, vocab_len):
        self.word_map = self.load_vocab(vocab_path)
        self.datas = self.get_data(data_path)
        self.text_len = text_len
        self.label_len = label_len
        self.vocab_len = vocab_len
    
    def get_data(self, data_path):
        datas = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                document = json.loads(line.strip())
                datas.append(document)
        return datas

    def load_vocab(self, vocab_path):
        word_map = {}
        with open(vocab_path, "r") as f:
            lines = f.readlines()
            for idx in range(len(lines)):
                word_map[lines[idx].strip()] = idx
        return word_map        
    
    def to_ids(self, inputs):
        output = []
        inputs = list(jieba.cut(inputs, cut_all=True))
        for key in inputs:
            if key in self.word_map:
                if self.word_map[key] < self.vocab_len:
                    output.append(self.word_map[key])
        return output
    
    def padding_feature(self, inputs, lenth):
        while len(inputs) < lenth:
            inputs.append(0)
        if len(inputs) > lenth:
            inputs = inputs[: lenth]
        return inputs
    
    def sparse_label(self, inputs):
        outputs = [0] * self.label_len
        for idx in range(len(inputs)):
            outputs[inputs[idx]] = 1
        return outputs


    def encoding_data(self):
        examples = []
        for data in self.datas:
            text = ",".join(data["paragraphs"])
            title = ",".join(data["title"])
            text_ids = self.to_ids(text)
            title_ids = self.to_ids(title)
            if len(text_ids) == 0 :
                continue
            example = {}
            example["feature"] = self.padding_feature(text_ids, self.text_len)
            example["label"] = self.sparse_label(random_list(3))
            example["id"] = data["id"]
            examples.append(example)
        return examples

class Evaluate(K.callbacks.Callback):
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
    
    def on_epoch_end(self, epoch, logs = None):
        self.model.save(self.save_dir)


class CuModel():
    # 模型运行方法
    def __init__(self, text_len, label_len, vocab_len):
        # text_len: 输入文本text长度截断
        # label_len: label 分类个数
        # vocab_len: 词表长度
        self.text_len = text_len
        self.label_len = label_len
        self.vocab_len = vocab_len

    def train(self, examples, save_path, batch_size, epoch_num):
        model = self.generate_cu_model()
        evaluator = Evaluate(model, save_path)
        steps_per_epoch = math.floor(len(examples) / (batch_size * epoch_num))
        data = self.data_generator(examples, batch_size)
        model.fit_generator(data,
                            steps_per_epoch = steps_per_epoch, 
                            epochs = epoch_num, 
                            shuffle = True, 
                            callbacks = [evaluator])
        K.backend.clear_session()
        return 

    def data_generator(self, examples, batch_size):
        inputs_key = ["feature", "label"]
        examples_nums = len(examples)
        while True:
            for idx in range(0, examples_nums, batch_size):
                batch_data = examples[idx: idx + batch_size]
                batch_map = {}
                for key in inputs_key:
                    batch_map[key] = []
                for per_example in batch_data:
                    for key in inputs_key:
                        batch_map[key].append(per_example[key])
                for key in inputs_key:
                    batch_map[key] = np.array(batch_map[key], dtype=object).astype('float32')
                yield (batch_map["feature"], batch_map["label"])
    
    def cross_entropy_loss(self, y_true, y_pred):
        y_pred = tf.math.log(y_pred)
        loss = tf.reduce_mean(-y_true * y_pred)
        return loss
    
    def generate_cu_model(self):
        feature = K.Input(name = 'feature', shape = [self.text_len])
        initializers = K.initializers.RandomNormal(stddev=0.02, mean = 0.0)
        feature_emb_layer = K.layers.Embedding(self.vocab_len, 64,
                                               input_length = self.text_len,
                                               embeddings_initializer = initializers,
                                               name = "feature_embedding" )
        attetnion = tf.keras.layers.Attention()
        fft_layer = K.layers.Dense(64, activation = "relu",
                                   kernel_initializer = initializers,
                                   bias_initializer = K.initializers.Zeros(),
                                   name = "fft1")
        cls_layer = K.layers.Dense(self.label_len, activation = "softmax",
                            kernel_initializer = initializers,
                            bias_initializer = K.initializers.Zeros(),
                            name = "cls_layer")
        ## process
        feature_emb = feature_emb_layer(feature)
        feature_emb_att = attetnion([feature_emb, feature_emb])
        feature_emb_ftt = fft_layer(feature_emb_att)
        feature_emb_ftt_shape = feature_emb_ftt.get_shape().as_list()
        feature_emb_ftt = tf.reshape(feature_emb_ftt, [-1, feature_emb_ftt_shape[1] * feature_emb_ftt_shape[2]])
        feature_cls = cls_layer(feature_emb_ftt)

        model = K.Model(inputs = [feature], outputs=[feature_cls])
        print(model.summary())
        opt = K.optimizers.Adam(learning_rate = 0.000001)
        model.compile(optimizer = opt, loss = self.cross_entropy_loss)
        return model
    
    def load_model(self, path):
        objects_fuc = {}
        objects_fuc["cross_entropy_loss"] = self.cross_entropy_loss
        if os.path.exists(path):
            model = K.models.load_model(path, custom_objects = objects_fuc)
            return model
        else:
            raise FileNotFoundError("model path is not exist : %s", path)

    def generator_predict_data(self, examples, batch_size):
        inputs_key = ["feature", "label", "id"]
        examples_nums = len(examples)
        output_datas = []
        for idx in range(0, examples_nums, batch_size):
            batch_data = examples[idx: idx + batch_size]
            batch_map = {}
            for key in inputs_key:
                batch_map[key] = []
            for per_example in batch_data:
                for key in inputs_key:
                    batch_map[key].append(per_example[key])
            for key in inputs_key:
                batch_map[key] = np.array(batch_map[key], dtype=object).astype('float32')
            output_datas.append(batch_map)
        return output_datas

    def anlysis_predict_data(self, predicts, cate_list):
        cates = []
        for predict in predicts:
            if len(predict) != len(cate_list):
                raise TypeError
            output = []
            for idx, score in enumerate(predict):
                if score > 0.01:
                    output.append(cate_list[idx])
            if len(output) < 3:
                top_k = heapq.nlargest(3, range(len(predict)), predict.__getitem__)
                for idx in top_k:
                    output.append(cate_list[idx])
            cates.append(output[:10])
        return  cates
    
    def load_cate_vocab(self, cate_vocab_path):
        cate_dic = {}
        with open(cate_vocab_path, "r") as f:
            lines = f.readlines()
            n = 0
            for line in lines:
                line = line.strip()
                cate_dic[n] = line
                n += 1
        return cate_dic

    def predict(self, model, examples, batch_size, cate_vocab_path):
        predict_data = self.generator_predict_data(examples, batch_size)
        cu_att_data = []
        cate_dict = self.load_cate_vocab(cate_vocab_path)
        for data in predict_data:
            scores = model.predict(data["feature"])
            cate_list = self.anlysis_predict_data(scores, cate_dict)
            if len(cate_list) != len(data["id"]):
                raise BoundaryError("len(cate_list) != len(data[id])")
            for idx in range(len(data["id"])):
                id = int(data["id"][idx])
                cu_dict = {}
                cu_dict["id"] = id
                cu_dict["cate"] = cate_list[idx]
                cu_att_data.append(cu_dict)
        return cu_att_data


class ContentUnderstandingAlgorithmContractExample(ContentUnderstandingAlgorithmContract):
    def __init__(self):
        # developer 定义
        self.model_ar_dir = "./models/cu_model/test"
        # text_len: 输入文本text长度截断
        # label_len: label 分类个数
        # vocab_len: 词表长度
        # vocab_path : 模型输入的词表地址
        # predict_batch_size : 预测的时候的batch size 大小
        # cate_vocab_path : 模型输出的类目对应的词表
        # save_cu_att_data_path : 存放生成的正排数据的本地地址
        self.text_len = 20
        self.label_len = 100
        self.vocab_len = 260000
        self.vocab_path = "./data/vocab"
        self.predict_batch_size = 10
        self.cate_vocab_path = "./data/cate_vocab"
        self.save_cu_att_data_path = "./data/forward.json"

    
    def parse(self, raw_content_path, model_local_dir):
        # load model
        model = CuModel(self.text_len, self.label_len, self.vocab_len)
        cu_model = model.load_model(model_local_dir)

        # 处理item的原始数据
        data_reader = DataReader(raw_content_path, self.vocab_path, self.text_len, self.label_len, self.vocab_len)
        examples = data_reader.encoding_data()

        # 预测每个item的类目并处理成输出所需要的格式
        cu_att_data = model.predict(cu_model, examples, self.predict_batch_size, self.cate_vocab_path)

        # 将数据，写入本地文件
        list_dict_json_file(cu_att_data, self.save_cu_att_data_path)
        return self.save_cu_att_data_path



def main():
    # raw_content_path item 原始item信息数据存放的本地地址
    raw_content_path = "./data/items.json"
    contract = ContentUnderstandingAlgorithmContractExample()
    model_ar_dir = contract.model_ar_dir
    model_local_dir = get_ar_file(model_ar_dir)
    forward_load_path = contract.parse(raw_content_path, model_local_dir)



# 训练模型
def train():
    text_len = 20
    label_len = 100
    vocab_len = 260000
    file_path = "./data/items.json"
    vocab_path = "./data/vocab"
    data_reader = DataReader(file_path, vocab_path, text_len, label_len, vocab_len)
    examples = data_reader.encoding_data()
    model = CuModel(text_len, label_len, vocab_len)
    save_path = "./models/cu_model/test"
    batch_size = 24
    epoch_num = 1
    model.train(examples, save_path, batch_size, epoch_num)
    return 

if __name__ == "__main__":
    train()


