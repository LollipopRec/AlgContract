import json
from algorithm_contract import UserContract
class UserContractExample(UserContract):
    def update_user_interest(self, user_raw_feature):
        user_cate_map = []
        for doc in user_raw_feature:
            if doc["click"] == 0:
                continue
            for cate in doc["read_ids_cate"]:
                if cate not in user_cate_map:
                    user_cate_map.append(cate)
        user_feature = {}
        user_feature["cate"] = user_cate_map
        return user_feature
