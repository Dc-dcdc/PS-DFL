import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated arguments
    parser.add_argument('--algorithm', type=str, default='PS_DFL', help="PS_DFL、PS_DFL_MC")
    parser.add_argument('--sampling', type=str, default='noniid', help="sampling method")
    parser.add_argument('--distribution', type=str, default='Dirichlet', help="1: Dirichlet  2: Manual-Division")
    parser.add_argument('--alpha', type=float, default=0.1, help='The smaller the alpha, the greater the difference in data distribution')
    parser.add_argument('--rounds', type=int, default=501, help="total number of communication rounds")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--num_groups', type=int, default=5, help="number of clusters: I")
    parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1, help="client learning rate")
    parser.add_argument('--lr_model', type=str, default='Decay', help="1: Decay  2: Fix  select learning model")
    parser.add_argument('--num_classes', type=int, default=7, help="number of classes")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--min_le', type=int, default=5, help="minimum number of local epoch")
    parser.add_argument('--max_le', type=int, default=15, help="maximum number of minimum local epoch")
    parser.add_argument('--frac', type=float, default=0.1, help="fraction of clients: C")

    # other arguments
    parser.add_argument('--seed', type=int, default=62, help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=-1, help="0 for GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--classwise', type=int, default=1500, help="number of images for each class (global dataset)")
    parser.add_argument('--num_channels', type=int, default=3 , help="number of channels of images")
    parser.add_argument('--verbose', action='store_false', help='verbose print')
    parser.add_argument('--all_clients', action='store_false', help='aggregation over all clients')
    parser.add_argument('--sys_homo', action='store_false', help='no system heterogeneity')
    parser.add_argument('--tsboard', action='store_true', help='tensorboard')
    parser.add_argument('--debug', action='store_false', help='debug')
    parser.add_argument('--title', type=int, default=1, help='decide the title of excel')
    args = parser.parse_args()
    
    return args
