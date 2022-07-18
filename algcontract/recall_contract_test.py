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
    user_feature = {'cate': {'misc.forsale': 0.18, 'sci.med': 0.08, 'comp.graphics': 0.02, 'talk.politics.guns': 0.18, 'rec.sport.hockey': 0.12, 'soc.religion.christian': 0.08, 'rec.autos': 0.06, 'comp.windows.x': 0.12, 'sci.crypt': 0.08, 'rec.sport.baseball': 0.05, 'comp.sys.mac.hardware': 0.02, 'talk.politics.misc': 0.02, 'comp.os.ms-windows.misc': 0.02}}
    recall_list = recall_contract.recall(user_feature, 20)
    print(recall_list)

recall_recall_test()
