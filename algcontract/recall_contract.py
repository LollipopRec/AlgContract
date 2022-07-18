import json
import numpy as np
import tensorflow.keras as K
import tensorflow as tf
import os
import math
import heapq
from algorithm_contract import RecallAlgorithmContract

# fake 接口, 从arweave上拉取数据，返回本地文件地址
def get_ar_file(ar_ads):
    ar_local_ads = ar_ads
    return ar_local_ads


class Evaluate(K.callbacks.Callback):
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
    
    def on_epoch_end(self, epoch, logs = None):
        self.model.save_weights(self.save_dir)


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
    
    def load_click_data(self, sample_dir_path, raw_content_path):
        # load  训练日志（点击日志）
        # 输入
        #    sample_dir_path: 训练日志本地地址
        #    raw_content_path 原始item信息
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
    
    def load_item_data(self, forward_local_dir):
        # 将正排本地文件，转成模型可接收的数据格式
        # 输入
        #    forward_local_dir: 正排本地地址
        # 输出
        #    item encoding list
        items = []
        with open(forward_local_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                items.append(line)
        return  self.encoding_item_data(items)
                
    def encoding_item_data(self, items):
        # 将item（dict 类型） 转成 encoding 完的 item 样本 -> item model 可接收的格式
        items_ids = []
        for item in items:
            item["doc_ids"] = self.cate_to_ids(item["cate"])
            item["doc_ids"] = self.padding_truncating_encoding(item["doc_ids"], self.doc_id_len)
            del item["cate"]
            items_ids.append(item)
        return items_ids
    
    def trans_user_encode(self, inputs):
        outputs = []
        for key in inputs:
            outputs.append(key)
        return outputs

    def encoding_data(self, examples):
        # 将模型训练用的样本，转换成 main model 可接收的输入格式
        examples_ids = []
        for example in examples:
            user_ids = self.cate_to_ids(self.trans_user_encode(example["user_feature"]["cate"]))
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
    
    def load_user_data(self, user_feature):
        # 将用户特征转换成 user_embeding_model 可以接受的输入格式
        user_encode = {}
        user_ids = self.cate_to_ids(self.trans_user_encode(user_feature))
        user_ids = self.padding_truncating_encoding(user_ids, self.user_id_len)
        user_encode["user_ids"] = user_ids
        return user_encode


class RecallModelExample(): 
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
        self.main_model_dir = "/main_model/"
        self.item_model_dir = "/item_model/"
        self.user_model_dir = "/user_model/"
        self.index_one_filename = "index_one.json"

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

    def generate_recall_model(self, mode):
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
        if mode == "train":
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
            cos_similarity = tf.reduce_sum(tf.multiply(doc_fc2,user_fc2))
            model = K.Model(inputs = [doc_feature, user_feature], outputs=[cos_similarity])
            opt = K.optimizers.Adam(learning_rate = 0.000001)
            model.compile(optimizer = opt, loss = self.bi_cross_entropy_loss)
        elif mode == "user":
            user_feature = K.Input(name = 'user_ids', shape = [self.user_id_len])
            user_emb = user_emb_layer(user_feature)
            user_fc1 = user_fc_layer_one(user_emb)
            user_fc1_dc = tf.squeeze(tf.nn.max_pool(user_fc1, ksize = [1, self.user_id_len, 1], strides = self.user_id_len, padding = "SAME"), 1)
            user_fc2 = user_fc_layer_two(user_fc1_dc)
            model = K.Model(inputs = [user_feature], outputs=[user_fc2])
        elif mode == "item":
            doc_feature = K.Input(name = 'doc_ids', shape = [self.doc_id_len])
            doc_emb = doc_emb_layer(doc_feature)
            doc_fc1 = doc_fc_layer_one(doc_emb)
            doc_fc1_dc = tf.squeeze(tf.nn.max_pool(doc_fc1, ksize = [1, self.doc_id_len, 1], strides = self.doc_id_len, padding= "SAME"), 1)
            doc_fc2 = doc_fc_layer_two(doc_fc1_dc)
            model = K.Model(inputs = [doc_feature], outputs=[doc_fc2])
        return model
    
    def bi_cross_entropy_loss(self, y_true, y_pred):
        # binary 交叉熵loss
        loss = - y_true * tf.math.log(y_pred) - (1- y_true) * tf.math.log(1 - y_pred)
        loss = tf.reduce_mean(loss)
        return loss
    
    def load_model_weight(self, path, model):
        # 加载模型权重
        if os.path.exists(path):
            model.load_weights(path)
            return model
        else:
            raise FileNotFoundError("model path is not exist : %s" % path)
    
    def train(self, examples, save_path, batch_size, epoch_num, old_model_dir = None):
        K.backend.clear_session()
        main_model_save_path = save_path + self.main_model_dir
        item_model_save_path = save_path + self.item_model_dir
        user_model_save_path = save_path + self.user_model_dir
        model = self.generate_recall_model("train")
        if old_model_dir != None:
            old_model_dir += self.main_model_dir
            model = self.load_model_weight(old_model_dir, model)
        evaluator = Evaluate(model, main_model_save_path)
        steps_per_epoch = math.floor(len(examples) / (batch_size * epoch_num))
        data = self.data_generator(examples, batch_size)
        model.fit_generator(data,
                            steps_per_epoch = steps_per_epoch, 
                            epochs = epoch_num, 
                            shuffle = True, 
                            callbacks = [evaluator])
        K.backend.clear_session()
        self.save_split_model(main_model_save_path, user_model_save_path, "user")
        self.save_split_model(main_model_save_path, item_model_save_path, "item")
    
    def save_split_model(self, main_model_save_path, model_save_path, mode):
        user_model = self.generate_recall_model(mode)
        user_model.load_weights(main_model_save_path)
        user_model.save(model_save_path)
        K.backend.clear_session()
    
    def item_predict(self, item_examples, model_local_dir, item_predict_batch_size):
        # 调用item embedding model 将item embedding
        item_model_path = model_local_dir + self.item_model_dir
        item_model = self.load_model(item_model_path)
        datas = self.item_predict_data_process(item_examples, item_predict_batch_size)
        items_emb = []
        for  data in datas:
            item_embedding = item_model.predict(data)
            items_emb += self.anlysis_item_data(item_embedding, list(data["id"]))
        return items_emb
    
    def check_path(self, path):
        if os.path.exists(path):
            return 
        else:
            os.mkdir(path)
    
    def build_item_index(self, item_examples, model_local_dir, item_predict_batch_size, index_local_save_path):
        # build 向量索引
        # 输入
        #    item_examples: 编码好的item样本
        #    model_local_dir: 模型文件的本地存储路径
        #    item_predict_batch_size: batch size
        #    index_local_save_path : 生成的索引文件的保存地址
        items_emb = self.item_predict(item_examples, model_local_dir, item_predict_batch_size)
        path = index_local_save_path + self.index_one_filename
        self.check_path(index_local_save_path)
        with open(path, "w") as f:
            for item_emb in items_emb:
                f.write(item_emb + "\n")

    def anlysis_item_data(self, item_embedding, ids):
        # 解析item embedding model的输出
        if len(item_embedding) != len(ids):
            raise IndexError("item_embedding length (%d) and ids length (%d) is not equal" % (len(item_embedding), len(ids)))
        items_emb = []
        for idx in range(len(ids)):
            item_emb = {}
            item_emb["id"] = int(ids[idx])
            item_emb["embed"] = ",".join(map(str, item_embedding[idx].tolist()))
            line = json.dumps(item_emb, ensure_ascii=False)
            items_emb.append(line)
        return items_emb

    def load_model(self, path):
        # 依赖pb文件，加载模型
        objects_fuc = {}
        objects_fuc["bi_cross_entropy_loss"] = objects_fuc
        if os.path.exists(path):
            model = K.models.load_model(path, custom_objects = objects_fuc)
            return model
        else:
            raise FileNotFoundError("model path is not exist : %s" % path)
    
    def item_predict_data_process(self, item_examples, item_predict_batch_size):
        # 将item model 的输入的example 处理成batch 的格式
        inputs_key = ["doc_ids", "id"]
        batch_predict_datas = []
        for idx in range(0, len(item_examples), item_predict_batch_size):
            batch_data = item_examples[idx: idx + item_predict_batch_size]
            batch_map = {}
            for key in inputs_key:
                batch_map[key] = []
            for per_example in batch_data:
                for key in inputs_key:
                    batch_map[key].append(per_example[key])
            for key in inputs_key:
                batch_map[key] = np.array(batch_map[key], dtype=object).astype('float32')
            batch_predict_datas.append(batch_map)
        return batch_predict_datas
    
    def load_item_index(self, index_local_dir):
        # 加载索引文件
        path = index_local_dir + self.index_one_filename 
        if not os.path.exists(path):
            raise FileNotFoundError("index local file is not exists : %s" % path)
        self.embs_list = []
        self.ids_list = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                item_emb = json.loads(line.strip())
                self.embs_list.append(item_emb["embed"].split(","))
                self.ids_list.append(item_emb["id"])
        self.embs_list = np.array(self.embs_list, dtype=object).astype('float32')
    
    def load_user_model(self, model_local_dir):
        # 加载user embedding model
        user_model_path = model_local_dir + self.user_model_dir
        self.user_model = self.load_model(user_model_path)
    
    def predict_sim(self, user_encode, top_nums):
        # 计算user embedding 和 item embedding的相似度
        input = {}
        input["user_ids"] = np.array([user_encode["user_ids"]], dtype=object).astype('float32')
        user_emb = self.user_model.predict(input)
        user_emb = user_emb.transpose()
        scores = np.dot(self.embs_list, user_emb)
        scores = list(np.squeeze(scores, axis = 1))
        top_k = heapq.nlargest(top_nums, range(len(scores)), scores.__getitem__)
        recall_list = []
        index_len = len(self.ids_list)
        for idx in top_k:
            if idx >= index_len:
                raise IndexError("recall idx (%d)is out of index range (%d)" % (idx, index_len))
            lollipop_item = {}
            lollipop_item["id"] = self.ids_list[idx]
            recall_list.append(lollipop_item)
        return recall_list


class RecallAlgorithmContractExample(RecallAlgorithmContract):
    def __init__(self, index_local_dir = None, model_local_dir = None):
        super(RecallAlgorithmContractExample, self).__init__(index_local_dir, model_local_dir)
        # 初始化继承部分      
        # developer  自定义部分
        # cate_vocab_path: 内容理解的类目词表，由 dapp 或者 init developer 提供, miner 记录在链上的一份文件
        # save_path: 训练模型本地导出目录
        # index_local_save_path: 生成的索引文件的本地存储地址
        self.cate_vocab_path = "./data/cate_vocab"
        self.save_path = "./models/recall_model/new_model/"
        self.index_local_save_path = "./data/index_dir/"

        # 算法参数
        # doc_id_len doc encoding 长度
        # user_id_len user encoding 长度
        # cate_vocab_len : 类目词表的长度
        doc_id_len = 10
        user_id_len = 10
        cate_vocab_len = 20

        self.batch_size = 64
        self.epoch_num = 1
        self.item_predict_batch_size = 128
        self.model = RecallModelExample(doc_id_len, user_id_len, cate_vocab_len)
        self.data_reader = DataReader(self.cate_vocab_path, doc_id_len, user_id_len)
        if index_local_dir != None and model_local_dir != None:
            self.load_index_and_user_model(index_local_dir, model_local_dir)


    def train(self, sample_dir_path, last_model_local_dir = None, raw_content_path = None):
        examples = self.data_reader.load_click_data(sample_dir_path, raw_content_path)
        self.model.train(examples, self.save_path, self.batch_size, self.epoch_num, last_model_local_dir)
        return self.save_path
    
    def build_index(self, forward_local_dir, model_local_dir):
        item_examples = self.data_reader.load_item_data(forward_local_dir)
        self.model.build_item_index(item_examples, model_local_dir, self.item_predict_batch_size, self.index_local_save_path)
        return self.index_local_save_path
    
    def load_index_and_user_model(self, index_local_dir, model_local_dir):
        self.model.load_item_index(index_local_dir)
        self.model.load_user_model(model_local_dir)
    
    def recall(self, user_features, item_nums = 300):
        user_encode = self.data_reader.load_user_data(user_features)
        items = self.model.predict_sim(user_encode, item_nums)
        return items


