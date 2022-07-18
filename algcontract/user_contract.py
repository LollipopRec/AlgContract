import json
from algorithm_contract import UserContract
class UserContractExample(UserContract):
    def update_user_interest(self, user_raw_feature):
        user_cate_map = {}
        all_cate = 0
        for doc in user_raw_feature:
            if doc["click"] == 0:
                continue
            for cate in doc["read_ids_cate"]:
                all_cate += 1
                if cate not in user_cate_map:
                    user_cate_map[cate] = 1
                else:
                    user_cate_map[cate] += 1
        user_feature = {}
        for c in user_cate_map:
            user_cate_map[c] = round(user_cate_map[c]/float(all_cate), 2)
        user_feature["cate"] = user_cate_map
        return user_feature
