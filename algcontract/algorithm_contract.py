class AlgorithmContract(object):
    def destroy(self):
        return


class UserContract(AlgorithmContract):
    # 功能：生产用户兴趣特征
    # 输入：
    #     item_consu  用户原始特征
    # 输出：
    #     用户兴趣
    def update_user_interest(self, user_raw_feature):
        return 


class RecallAlgorithmContract(AlgorithmContract):
    def __init__(self, index_local_dir, model_local_dir):
        super(RecallAlgorithmContract, self).__init__()

    # 功能：获取经过Rank后的Item列表
    # 输入：
    #     recall_items  召回返回的Item列表
    #     user_features 用户侧特征
    # 输出：
    #     经排序后的Item列表
    def train(self, samples_local_dir, last_model_local_dir):
        return 

    # 输入：
    #     forward_local_dir: 正排索引文件的地址
    #     model_local_dir: 模型文件的地址
    # 输出:
    #     build的index的本地文件的地址
    def build_index(self, forward_local_dir, model_local_dir):   
        return

    # 输入
    #    user_features: 用户特征
    #    item_nums: 召回个数
    # 输出
    #    召回的item list
    def recall(self, user_features, item_nums = 300):
        return 


class ContentUnderstandingAlgorithmContract(AlgorithmContract):
    def __init__(self):
        super(ContentUnderstandingAlgorithmContract, self).__init__()
        return

    # 功能：对原始Item进行理解
    # 输入：
    #     raw_content  原始内容
    # 输出：
    #     内容理解结果（正排结构）
    def parse(self, raw_content_path, model_local_dir):
        return 


class RankAlgorithmContract(AlgorithmContract):
    def __init__(self, model_local_dirs = None):
        super(RankAlgorithmContract, self).__init__()
        # model_ar_dir 由developer初始化，model trainer 修改
    
    
    # 功能：Rank模型训练
    # 输入：
    #     samples_local_dir  保存样本的本地目录
    #     last_model_local_dir  上一版本模型本地目录
    # 输出：
    #     新模型导出的本地目录
    def train(self, samples_local_dir, last_model_local_dir):
        return 

    # 功能：获取经过Rank后的Item列表
    # 输入：
    #     recall_items  召回返回的Item列表
    #     user_features 用户侧特征
    # 输出：
    #     经排序后的Item列表
    def rank(self, recall_items, user_features, forward_data, reserve_num = 10):
        return 
    
  
   
   
   




