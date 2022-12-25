import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')

    parser.add_argument('--exp_name', type=str, default='gcn',
                        help='experiment name')

    parser.add_argument('--graph_name', type=str, default='Cora',
                        help='dataset name')
    parser.add_argument('--hidden_layer_num', type=int, default=2,
                        help='GCN hidden layer num')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='GCN hidden layer dimension')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='train:val split')
    parser.add_argument('--sample_neighbor_num', type=int, default=5,
                        help='sample_neighbor_num')

                        


    return parser.parse_args()