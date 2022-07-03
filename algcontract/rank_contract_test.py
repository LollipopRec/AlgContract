from rank_contract import RankAlgorithmContractExample
import json

def train_init_test():
    sample_dir_path = "./data/click_log.json"
    contract = RankAlgorithmContractExample()
    contract.train(sample_dir_path)


def train_daliy_test():
    sample_dir_path = "./data/click_log.json"
    model_ar_dir = "./models/rank_model/init_model/"
    contract = RankAlgorithmContractExample()
    contract.train(sample_dir_path, model_ar_dir)

def rank_init_test():
    model_ar_dir = "./models/rank_model/init_model/"
    contract = RankAlgorithmContractExample(model_ar_dir)
    return contract


def load_forward(forward_local_dir):
    forward_items = {}
    with open(forward_local_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            id = line["id"]
            del line["id"]
            forward_items[id] = line
    return forward_items

def rank_rank_test():
    contract = rank_init_test()
    recall_items = [{'id': 263}, {'id': 720}, {'id': 882}, {'id': 241}, {'id': 383}, {'id': 658}, {'id': 380}, {'id': 381}, {'id': 438}, {'id': 687}, {'id': 479}, {'id': 778}, {'id': 627}, {'id': 465}, {'id': 351}, {'id': 283}, {'id': 721}, {'id': 499}, {'id': 712}, {'id': 591}]
    user_features = {'cate': ['宠物零食', '影视', '游戏', '游戏名词', '游戏名', '角色名', '游戏解说', '游戏比赛', '体育', '运动教学', '电视剧', '纪录片', '奖项', '游戏资讯', '运动品牌', '少儿', '体育资讯', '动漫', '体育明星', '运动项目', '电影', '组织', '民间/大众运动', '综艺']}
    forward_local_dir = "./data/forward.json"
    forward_items = load_forward(forward_local_dir)
    print(contract.rank(recall_items, user_features, forward_items, reserve_num = 10))

