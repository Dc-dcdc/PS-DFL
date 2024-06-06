import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
from datetime import datetime
def iid(dataset, num_users):

    num_items = int(len(dataset)/(num_users))
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):

        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users

def noniid(dataset, train_labels,args):

#-----------------------------------------------------------------------Dirichlet分割数据集_START-----------------------------------------------------------------
    if args.distribution == 'Dirichlet' :

        # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
        label_distribution = np.random.dirichlet([args.alpha] * args.num_users, args.num_classes)
        # (K, ...) 记录K个类别对应的样本索引集合
        class_idcs = [np.argwhere(train_labels == y).flatten()
                      for y in range(args.num_classes)]

        # 记录N个client分别对应的样本索引集合
        dict_users = [[] for _ in range(args.num_users)]
        for k_idcs, fracs in zip(class_idcs, label_distribution):
            # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
            # i表示第i个client，idcs表示其对应的样本索引集合idcs
            for i, idcs in enumerate(np.split(k_idcs,(np.cumsum(fracs)[:-1]*len(k_idcs)).astype(int))):
                dict_users[i] += [idcs]


        dict_users = [np.concatenate(idcs) for idcs in dict_users]   #每个客户端的拥有的数据



#-----------------------------------------------------------------------Dirichlet分割数据集_END-----------------------------------------------------------------
################################################################################################################################################################
# -----------------------------------------------------------------------手动分割数据集_START----------------------------------------------------------------------------
    elif args.distribution == 'Manual-Division':
        num_dataset = len(dataset)

        idx = np.arange(num_dataset)

        dict_users = [[] for _ in range(args.num_users)]
        dict_users[0] = idx[0:300]    #2800
        dict_users[1] = idx[300:600]  # 1400
        dict_users[2] = idx[600:900]  # 1800
        dict_users[3] = idx[900:1200]  # 2700
        dict_users[4] = idx[1200:1473]  # 3200
        dict_users[5] = idx[1473:1746]  # 2100
        dict_users[6] = idx[1746:2019]  # 900
        dict_users[7] = idx[2019:2292]  # 1600
        dict_users[8] = idx[2292:2594]  # 2500
        dict_users[9] = idx[2594:2896]  # 2800
        dict_users[10] = idx[2896:3197]  # 1400
        dict_users[11] = idx[3197:3548]  # 1400
        dict_users[12] = idx[3548:3899]  # 1800
        dict_users[13] = idx[3899:4250]  # 2700
        dict_users[14] = idx[4250:4738]  # 3200
        dict_users[15] = idx[4738:5226]  # 2100
        dict_users[16] = idx[5226:5766]  # 900
        dict_users[17] = idx[5766:6306]  # 1600
        dict_users[18] = idx[6306:6816]  # 2500
        dict_users[19] = idx[6816:7326]  # 3600

# -----------------------------------------------------------------------手动分割数据集_END-----------------------------------------------------------------
################################################################################################################################################################
# -----------------------------------------------------------------------数据分布可视化_START-----------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(args.num_classes)]
    for c_id, idc in enumerate(dict_users):
        for idx in idc:
            label_distribution[train_labels[idx]].append(c_id)


    plt.hist(label_distribution, stacked=True, bins=np.arange(-0.5, args.num_users , 1),label=[i for i in range(args.num_classes)], rwidth=0.5)
    plt.xticks(np.arange(args.num_users), ["Client %d" % c_id for c_id in range(args.num_users)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    # plt.show()
    dt = datetime.now()
    if args.distribution == 'Dirichlet':
        plt.savefig(f'fig/num_users={args.num_users}_Non-iid_alpha={args.alpha}_{args.seed}_{dt.date()}-{dt.hour}h.png')
    else:
        plt.savefig(f'fig/num_users={args.num_users}_Non-iid_手动分_{dt.date()}-{dt.hour}h.png')

# -----------------------------------------------------------------------数据分布可视化_END-----------------------------------------------------------------
#     print(dict_users)
    return dict_users



def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs