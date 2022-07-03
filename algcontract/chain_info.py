
# fake 接口, 从arweave上拉取数据，返回本地文件地址
def get_ar_file(ar_ads):
    ar_local_ads = ar_ads
    return ar_local_ads


class OtherInfo():
    def __init__(self):
        self.rank_test_ar_ads = "./data/click_log.json"
        self.rank_metrics_contract_ads = ["fdsfssadfsdd"]


class ExpBucket():
    def __init__(self):
        self.recall_bucket = ["324de23edsqw", "xxfasfdadeds", "fdsfasdewacw"]    
        self.rank_bucket = ["fdsfsdfdsfds", "fdsafadsfcvf", "fdasfarefedf"]  
        self.cu_bucket = ["fdsiowqerkds", "fuiwnvwdjfew", "fiqweffwnuic"]     

class ContractState():
    def __init__(self, module_path = None, 
                 module_name = None,
                 model_ar_dir = None, 
                 index_ar_dir = None, 
                 forward_ar_dir = None,):
        self.module_path = module_path
        self.module_name = module_name
        self.model_ar_dir = model_ar_dir
        self.index_ar_dir = index_ar_dir
        self.forward_ar_dir = forward_ar_dir

def read_from_chain(address):
    contracts = {}
    # recall
    contracts["324de23edsqw"] =  ContractState(module_path = "./src/recall_contract.py", module_name = "RecallAlgorithmContractExample",
                                               model_ar_dir = "./models/recall_model/init_model/", index_ar_dir = "./data/index_dir/")
    contracts["xxfasfdadeds"] =  ContractState(module_path = "./src/recall_contract.py", module_name = "RecallAlgorithmContractExample",
                                               model_ar_dir = "./models/recall_model/init_model/", index_ar_dir = "./data/index_dir/")
    contracts["fdsfasdewacw"] =  ContractState(module_path = "./src/recall_contract.py", module_name = "RecallAlgorithmContractExample",
                                               model_ar_dir = "./models/recall_model/init_model/", index_ar_dir = "./data/index_dir/")

    # rank
    contracts["fdsfsdfdsfds"] =  ContractState(module_path = "./src/rank_contract.py", module_name = "RankAlgorithmContractExample",
                                               model_ar_dir = "./models/rank_model/init_model/")
    contracts["fdsafadsfcvf"] =  ContractState(module_path = "./src/rank_contract.py", module_name = "RankAlgorithmContractExample",
                                               model_ar_dir = "./models/rank_model/init_model/")
    contracts["fdasfarefedf"] =  ContractState(module_path = "./src/rank_contract.py", module_name = "RankAlgorithmContractExample",
                                               model_ar_dir = "./models/rank_model/init_model/")

    # cu
    contracts["fdsiowqerkds"] = ContractState(module_path = "./src/cu_contract.py", module_name = "ContentUnderstandingAlgorithmContractExample",
                                              forward_ar_dir = "./data/forward.json")
    contracts["fuiwnvwdjfew"] = ContractState(module_path = "./src/cu_contract.py", module_name = "ContentUnderstandingAlgorithmContractExample",
                                              forward_ar_dir = "./data/forward.json")
    contracts["fiqweffwnuic"] = ContractState(module_path = "./src/cu_contract.py", module_name = "ContentUnderstandingAlgorithmContractExample",
                                              forward_ar_dir = "./data/forward.json")
    

    # metrcis
    contracts["fdsfssadfsdd"] = ContractState(module_path = "./src/metrics.py", module_name = "RankMetricsExample")
    return contracts[address]