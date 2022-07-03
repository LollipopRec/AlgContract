from cu_contract import ContentUnderstandingAlgorithmContractExample

def parse_test():
    contract = ContentUnderstandingAlgorithmContractExample()
    model_ar_dir = "./models/cu_model/test"
    raw_content_path = "./data/items.json"
    forward_load_path = contract.parse(raw_content_path, model_ar_dir)
    print("forward_load_path", forward_load_path)

parse_test()