import json
import random

random_seed=1
random.seed(random_seed)

def get_indices():
    with open('./data/mashup_name.json', 'r') as file:
        dataset = json.load(file)
    # with open('./data/mashup_used_api.json', 'r') as f:
    #     mashup_used_api_ = json.load(f)

    # filter_idx = []
    # for idx, apis in enumerate(mashup_used_api_):
    #     # if len(apis) >= 2:
    #     filter_idx.append(idx)

    #data_idx = list(range(len(dataset)))
    # random.shuffle(dataset)
    split_num = int(len(dataset) / 10)
    test_idx = dataset[:split_num]
    train_idx = dataset[split_num:]
    print("len(train_idx), len(test_idx)----------------------", len(train_idx), len(test_idx))

    #train_idx, test_idx = train_test_split(dataset, test_size=0.3, random_state=random_seed)


    train_apis = set()
    oov_api = set()
    '''
    with open('./data/mashup_used_api.json', 'r') as f, open('./data/mashup_name.json', 'r') as f2:
        mashups = json.load(f2)
        mashup_apis = json.load(f)
        for idx, mashup in enumerate(mashups):
            if mashup in set(train_idx):
                apis = mashup_apis[idx]
                for api in apis:
                    train_apis.add(api)

    with open('./data/mashup_used_api.json', 'r') as f, open('./data/mashup_name.json', 'r') as f2:
        mashups = json.load(f2)
        mashup_apis = json.load(f)
        for idx, mashup in enumerate(mashups):
            if mashup in set(test_idx):
                apis = mashup_apis[idx]
                for api in apis:
                    if api not in train_apis:
                        print(api)
                        oov_api.add(api+'_api')
    '''
    print('oov {}'.format(len(oov_api)))
    return train_idx, test_idx, oov_api

