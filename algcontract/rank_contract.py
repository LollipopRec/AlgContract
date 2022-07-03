from email.errors import BoundaryError
import json
import numpy as np
import tensorflow.keras as K
import tensorflow as tf
import os
import math
from algorithm_contract import RankAlgorithmContract
from chain_info import read_from_chain

# fake 接口, 从arweave上拉取数据，返回本地文件地址
def get_ar_file(ar_ads):
    ar_local_ads = ar_ads
    return ar_local_ads

class DataReader(object):
    # 数据处理流程
    def __init__(self, cate_vocab_path, doc_id_len, user_id_len):
        # cate_vocab_path: 类目词表本地地址
        # doc_id_len: doc encoding 时候的截断长度
        # user_id_len: 用户特征encoding 时候的截断长度
        self.cate_dict = self.load_cate_vocab(cate_vocab_path)
        self.doc_id_len = doc_id_len
        self.user_id_len = user_id_len
    
    def load_cate_vocab(self, cate_vocab_path):
        # load 类目词表
        # 输入
        #    cate_vocab_path: 类目词表本地地址
        # 输出
        #    类目词表dict  dict[类目词] = id
        cate_dict = {}
        with open(cate_vocab_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                cate_dict[line.strip()] = idx
        return cate_dict
    
    def load_click_data(self, sample_dir_path):
        # load  训练日志（点击日志）
        # 输入
        #    sample_dir_path: 训练日志本地地址
        # 输出
        #    sample List: encoding 完的样本，组成的list
        examples = []
        with open(sample_dir_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                examples.append(line)
        examples_ids = self.encoding_data(examples)
        return examples_ids
    
    def prepare_predict_data(self, user_feature, forward_data, items):
        examples = []
        user_ids = self.cate_to_ids(user_feature)
        user_ids = self.padding_truncating_encoding(user_ids, self.user_id_len)
        for item in items:
            item_id = item["id"]
            doc_cate = forward_data[item_id]["cate"]
            doc_ids = self.cate_to_ids(doc_cate)
            doc_ids = self.padding_truncating_encoding(doc_ids, self.doc_id_len)
            example = {}
            example["user_ids"] = user_ids
            example["doc_ids"] = doc_ids
            examples.append(example)
        return examples

    def encoding_data(self, examples):
        # 将模型训练用的样本，转换成 main model 可接收的输入格式
        examples_ids = []
        for example in examples:
            user_ids = self.cate_to_ids(example["user_feature"]["cate"])
            user_ids = self.padding_truncating_encoding(user_ids, self.user_id_len)
            doc_samples = self.process_doc_sample(example["docs"])
            for sample in doc_samples:
                sample["user_ids"] = user_ids
                examples_ids.append(sample)
        return examples_ids
    
    def cate_to_ids(self, input_list):
        #  将类目词编码
        ids_list = []
        for key in input_list:
            if key in self.cate_dict:
                ids_list.append(self.cate_dict[key])
        return ids_list
    
    def padding_truncating_encoding(self, input_dict, padding_length):
        # 根据模型接收的长度，进行截断或者padding
        while len(input_dict) < padding_length:
            input_dict.append(0)
        return input_dict[: padding_length]
    
    def process_doc_sample(self, doc_samples):
        # 将训练样本中的，用户浏览过的item list 转换成一条条sample
        samples = []
        for doc_sample in doc_samples:
            sample = {}
            doc_encode = self.cate_to_ids(doc_sample["read_ids_cate"])
            doc_encode = self.padding_truncating_encoding(doc_encode, self.doc_id_len)
            label = doc_sample["click"]
            sample["label"] = label
            sample["doc_ids"] = doc_encode
            samples.append(sample)
        return samples
    
class Evaluate(K.callbacks.Callback):
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
    
    def on_epoch_end(self, epoch, logs = None):
        self.model.save(self.save_dir)

class RankModelExample(): 
    # 召回模型，模型方法
    def __init__(self, doc_id_len, user_id_len, cate_vocab_len):
        # doc_id_len: doc进行编码的时候的最大长度
        # user_id_len: 用户特征进行编码的时候的最大长度
        # cate_vocab_len: 类目词表的长度
        # main_model_dir: 双塔模型，模型参数权重存储地址
        # item_model_dir: item 编码模型pb文件存储地址
        # user_model_dir: user 编码模型pb文件存储地址
        # index_one_filename: 索引文件名之一
        self.doc_id_len = doc_id_len
        self.user_id_len = user_id_len
        self.cate_vocab_len = cate_vocab_len

    def data_generator(self, examples, batch_size):
        # 训练样本数据生产 generator
        # 输入
        #    examples: encoding 好的训练
        inputs_key = ["doc_ids", "user_ids" ,"label"]
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
                label = batch_map["label"]
                del batch_map["label"]
                yield (batch_map, label)

    def generate_recall_model(self):
        # 构造tf模型图
        initializers = K.initializers.RandomNormal(stddev=0.02, mean = 0.0)
        doc_emb_layer = K.layers.Embedding(self.cate_vocab_len, 64,
                                           input_length = self.doc_id_len,
                                           embeddings_initializer = initializers,
                                           name = "doc_embedding" )
        user_emb_layer = K.layers.Embedding(self.cate_vocab_len, 64,
                                            input_length = self.user_id_len,
                                            embeddings_initializer = initializers,
                                            name = "user_embedding" )
   
        doc_fc_layer_one = K.layers.Dense(64, activation = "relu",
                                          kernel_initializer = initializers,
                                          bias_initializer = K.initializers.Zeros(),
                                          name = "doc_fc1")
        doc_fc_layer_two = K.layers.Dense(64, activation = "relu",
                                          kernel_initializer = initializers,
                                          bias_initializer = K.initializers.Zeros(),
                                          name = "doc_fc2")
        user_fc_layer_one = K.layers.Dense(64, activation = "relu",
                                          kernel_initializer = initializers,
                                          bias_initializer = K.initializers.Zeros(),
                                          name = "user_fc1")
        user_fc_layer_two = K.layers.Dense(64, activation = "relu",
                                          kernel_initializer = initializers,
                                          bias_initializer = K.initializers.Zeros(),
                                          name = "user_fc2")
        ## process
        doc_feature = K.Input(name = 'doc_ids', shape = [self.doc_id_len])
        user_feature = K.Input(name = 'user_ids', shape = [self.user_id_len])
        doc_emb = doc_emb_layer(doc_feature)
        user_emb = user_emb_layer(user_feature)
        doc_fc1 = doc_fc_layer_one(doc_emb)
        doc_fc1_dc = tf.squeeze(tf.nn.max_pool(doc_fc1, ksize = [1, self.doc_id_len, 1], strides = self.doc_id_len, padding= "SAME"), 1)
        doc_fc2 = doc_fc_layer_two(doc_fc1_dc)
        user_fc1 = user_fc_layer_one(user_emb)
        user_fc1_dc = tf.squeeze(tf.nn.max_pool(user_fc1, ksize = [1, self.user_id_len, 1], strides = self.user_id_len, padding = "SAME"), 1)
        user_fc2 = user_fc_layer_two(user_fc1_dc)
        cos_similarity = tf.reduce_sum(tf.multiply(doc_fc2,user_fc2), 1, keepdims=True)
        model = K.Model(inputs = [doc_feature, user_feature], outputs=[cos_similarity])
        opt = K.optimizers.Adam(learning_rate = 0.000001)
        model.compile(optimizer = opt, loss = self.bi_cross_entropy_loss)
        return model
    
    def bi_cross_entropy_loss(self, y_true, y_pred):
        # binary 交叉熵loss
        loss = - y_true * tf.math.log(y_pred) - (1- y_true) * tf.math.log(1 - y_pred)
        loss = tf.reduce_mean(loss)
        return loss
    
    def train(self, examples, save_path, batch_size, epoch_num, old_model_dir = None):
        model = self.generate_recall_model()
        if old_model_dir != None:
            model = self.load_model(old_model_dir)
        evaluator = Evaluate(model, save_path)
        steps_per_epoch = math.floor(len(examples) / (batch_size * epoch_num))
        data = self.data_generator(examples, batch_size)
        model.fit_generator(data,
                            steps_per_epoch = steps_per_epoch, 
                            epochs = epoch_num, 
                            shuffle = True, 
                            callbacks = [evaluator])
        K.backend.clear_session()
    
    def load_model(self, path):
        # 依赖pb文件，加载模型
        objects_fuc = {}
        objects_fuc["bi_cross_entropy_loss"] = self.bi_cross_entropy_loss
        if os.path.exists(path):
            model = K.models.load_model(path, custom_objects = objects_fuc)
            return model
        else:
            raise FileNotFoundError("model path is not exist : %s" % path)
        

    def predict_data_generator(self, examples):
        inputs_key = ["doc_ids", "user_ids"]
        batch_map = {}
        for key in inputs_key:
            batch_map[key] = []
        for example in examples:
            for key in inputs_key:
                batch_map[key].append(example[key])
        for key in batch_map:
            batch_map[key] =  np.array(batch_map[key], dtype=object).astype('float32')
        return batch_map




class RankAlgorithmContractExample(RankAlgorithmContract):
    def __init__(self, model_local_dir = None):
        super(RankAlgorithmContractExample, self).__init__(model_local_dir)
        # cate_vocab_path: 内容理解的类目词表，由 dapp 或者 init developer 提供, miner 记录在链上的一份文件
        # save_path: 训练模型本地导出目录
        self.cate_vocab_path = "./data/cate_vocab"
        self.save_path = "./models/rank_model/new_model/"

        # 算法参数
        # doc_id_len doc encoding 长度
        # user_id_len user encoding 长度
        # cate_vocab_len : 类目词表的长度
        doc_id_len = 10
        user_id_len = 10
        cate_vocab_len = 100
        self.batch_size = 64
        self.epoch_num = 1

        self.data_reader = DataReader(self.cate_vocab_path, doc_id_len, user_id_len)
        self.model = RankModelExample(doc_id_len, user_id_len, cate_vocab_len)
        if model_local_dir != None:
            self.load_model(model_local_dir)
    
    def train(self, samples_local_dir, last_model_local_dir = None):
        examples = self.data_reader.load_click_data(samples_local_dir)
        self.model.train(examples, self.save_path, self.batch_size, self.epoch_num, last_model_local_dir)
        return self.save_path,
    
    def rank(self, recall_items, user_features, forward_data, reserve_num = 10):
        examples = self.data_reader.prepare_predict_data(user_features, forward_data, recall_items)
        data = self.model.predict_data_generator(examples)
        scores = self.model_func.predict(data)
        scores = list(np.transpose(scores)[0])
        if len(scores) != len(recall_items):
            raise BoundaryError("scores len : %d is not equal recall_items : %d" % (len(scores), len(recall_items)))
        for idx, item in enumerate(recall_items):
            item["rank_score"] = scores[idx]
            item["feature"] = forward_data[item["id"]]
        result = sorted(recall_items, key = lambda x: x["rank_score"], reverse = True)
        return result[: reserve_num]

    def load_model(self, model_local_dir):
        self.model_func = self.model.load_model(model_local_dir)
    

def train():
    sample_dir_path = "./data/click_log.json"
    contract = RankAlgorithmContractExample()
    contract.train(sample_dir_path)

train()