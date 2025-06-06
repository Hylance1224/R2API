import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/TE.csv'
        test_file = path + '/RS.csv'
        api_tag_file=path+'/api_tags.csv'
        mashup_tag_file=path+'/mashup_tags.csv'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.n_api_tag,self.n_mashup_tag=0,0
        self.n_api,self.n_mashup=0,0
        self.neg_pools = {}

        self.exist_users = []#去重后的用户，每个用户只在列表出现一次·

        #add a dictionary to store the recommendation result.
        self.recommendResult = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').strip().split(' ')
                    cl = [item for item in l if item != '']
                    # print(cl)
                    items = [int(i) for i in cl[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        with open(api_tag_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                        api=int(l.split(' ')[0])
                    except Exception:
                        continue
                    self.n_api_tag = max(self.n_api_tag, max(items))
                    self.n_api =max(self.n_api,api)
        with open(mashup_tag_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                        mashup = int(l.split(' ')[0])
                    except Exception:
                        continue
                    self.n_mashup_tag = max(self.n_mashup_tag, max(items))
                    self.n_mashup =max(self.n_mashup,mashup)
        self.n_mashup+=1
        self.n_api+=1
        self.n_mashup_tag+=1
        self.n_api_tag+=1


        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) #稀疏矩阵
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.R_Api=sp.dok_matrix((self.n_api,self.n_api_tag),dtype=np.float32)
        self.R_Mashup=sp.dok_matrix((self.n_mashup,self.n_mashup_tag),dtype=np.float32)


        self.train_items, self.test_set = {}, {}#添加训练集、测试集
        self.mashup_items,self.api_set={},{}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n').split(' ')
                    l = [item for item in l if item != '']
                    items = [int(i) for i in l]
                    uid, train_items = items[0], items[1:]

                    for idx, i in enumerate(train_items):
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n').split(' ')
                    l = [item for item in l if item != '']
                    items = [int(i) for i in l]

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        with open(mashup_tag_file) as f_mashup:
            for l in f_mashup.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]

                for idx, i in enumerate(train_items):
                    self.R_Mashup[uid, i] = 1.

                self.mashup_items[uid] = train_items
        with open(api_tag_file) as f_api:
            for l in f_api.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]

                for idx, i in enumerate(train_items):
                    self.R_Api[uid, i] = 1.

                self.api_set[uid] = train_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            #print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat) #+ sp.eye(adj_mat.shape[0])
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    #给用户生成一个负采样池，每个用户采样200个不相关的物品
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(200)]
            self.neg_pools[u] = pools

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)#random.sample()从指定序列中随机截取指定长度的序列
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
            #ramdom.choice()从指定序列中随机返回一个值
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]#返回的是一个列表，[0]取列表中的值
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_pos_tags_for_u(u, num):
            pos_tags = self.mashup_items[u]
            n_pos_tags = len(pos_tags)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_tags, size=1)[0]#返回的是一个列表，[0]取列表中的值
                pos_i_id = pos_tags[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_tags_for_u(u, num):
            neg_tags = []
            while True:
                if len(neg_tags) == num: break
                neg_id = np.random.randint(low=0, high=self.n_mashup_tag, size=1)[0]
                if neg_id not in self.mashup_items[u] and neg_id not in neg_tags:
                    neg_tags.append(neg_id)
            return neg_tags

        def sample_pos_tags_for_i(i, num):
            pos_tags = self.api_set[i]
            n_pos_tags = len(pos_tags)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_tags, size=1)[0]#返回的是一个列表，[0]取列表中的值
                pos_i_id = pos_tags[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_tags_for_i(i, num):
            neg_tags = []
            while True:
                if len(neg_tags) == num: break
                neg_id = np.random.randint(low=0, high=self.n_api_tag, size=1)[0]
                if neg_id not in self.api_set[i] and neg_id not in neg_tags:
                    neg_tags.append(neg_id)
            return neg_tags

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        pos_mashup_tags,neg_mashup_tags=[],[]
        pos_api_tags,neg_api_tags=[],[]
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)  #采样一个正相关的物品
            neg_items += sample_neg_items_for_u(u, 1)

            pos_mashup_tags += sample_pos_tags_for_u(u,1)
            neg_mashup_tags += sample_neg_tags_for_u(u, 1)

            pos_api_tags += sample_pos_tags_for_i(pos_items[-1], 1)
            neg_api_tags += sample_neg_tags_for_i(pos_items[-1], 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items,pos_mashup_tags,neg_mashup_tags,pos_api_tags,neg_api_tags


    def sample_all_users_pos_items(self):
        self.all_train_users = []
        self.all_train_pos_items = []
        for u in self.exist_users:
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        #为用户采样一定数量的负例物品
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u,1)

        perm = np.random.permutation(len(self.all_train_users))
        # 打乱训练用户、正例样本、负例样本
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    # print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')
        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
