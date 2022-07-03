from recall_contract import RecallAlgorithmContractExample

def recall_train_init():
    # train init model
    contract = RecallAlgorithmContractExample()
    sample_dir_path = "./data/click_log.json"
    local_save_path = contract.train(sample_dir_path)
    print("local_save_path %s" % local_save_path)

def recall_train_daliy():
    # train init model
    contract = RecallAlgorithmContractExample()
    sample_dir_path = "./data/click_log.json"
    old_model_path = "./models/recall_model/init_model/"
    local_save_path = contract.train(sample_dir_path, old_model_path)
    print("local_save_path %s" % local_save_path)

def recall_build_index_test():
    contract = RecallAlgorithmContractExample()
    forward_local_dir = "./data/forward.json"
    model_local_dir = "./models/recall_model/init_model/"
    contract.build_index(forward_local_dir, model_local_dir)


def recall_init_test():
    model_ar_dir = "./models/recall_model/init_model/"
    index_ar_dir = "./data/index_dir/"
    contract = RecallAlgorithmContractExample(index_ar_dir, model_ar_dir)
    return contract

def recall_recall_test():
    recall_contract = recall_init_test()
    user_feature = {'cate': ['宠物零食', '影视', '游戏', '游戏名词', '游戏名', '角色名', '游戏解说', '游戏比赛', '体育', '运动教学', '电视剧', '纪录片', '奖项', '游戏资讯', '运动品牌', '少儿', '体育资讯', '动漫', '体育明星', '运动项目', '电影', '组织', '民间/大众运动', '综艺']}
    recall_list = recall_contract.recall(user_feature, 20)
    print(recall_list)

if __name__ == "__main__":
    recall_recall_test()