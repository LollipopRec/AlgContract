from user_contract import UserContractExample
import random
import json

def loader(data_path):
    user_examples = []
    with open(data_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            user_examples.append(line["docs"])
    idx = random.randint(0, len(user_examples) - 1)
    return user_examples[idx]

def test():
    user_local_log_file = "./data/user_local_log.json"
    user_raw_feature = loader(user_local_log_file)
    user_contract = UserContractExample()
    print ("user_feature : %s" % user_contract.update_user_interest(user_raw_feature))

if __name__ == "__main__":
    test()
    