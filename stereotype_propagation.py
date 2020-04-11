import os
from utils import parse_command_line_args
args = parse_command_line_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu != -1 else ''

import pickle

from loguru import logger
import torch

tt = torch.cuda if torch.cuda.is_available() else torch


def normalize_trans_matrix(adjacency_matrix):
    out_degree_matrix = torch.diag(1 / torch.sqrt(adjacency_matrix.sum(dim=1)))
    in_degree_matrix = torch.diag(1 / torch.sqrt(adjacency_matrix.sum(dim=0)))
    trans_matrix = torch.chain_matmul(out_degree_matrix, adjacency_matrix, in_degree_matrix)
    return trans_matrix


def stereotype_propagation(trans_matrix, initial_matrix, alpha):
    """stereotype propagation: P^{*}=(1-\alpha)(I-\alpha T)^{-1} P_{0}"""
    identity_matrix = torch.eye(trans_matrix.size(0), device='cuda' if torch.cuda.is_available() else 'cpu')
    inv_trans_matrix = torch.inverse(identity_matrix - alpha * trans_matrix)
    information_matrix = (1 - alpha) * torch.matmul(inv_trans_matrix, initial_matrix)

    # calculate scores from information matrix
    information_matrix /= information_matrix.sum(dim=-1, keepdim=True)
    information_matrix = torch.log(information_matrix[:, 0] / information_matrix[:, 1])

    return information_matrix


def load_trans_matrix_from_graph(graph):
    vocab_dict = graph['vocab_dict']
    initial_matrix = graph['initial_matrix']
    adjacency_matrix = graph['adjacency_matrix']

    # into torch tensor
    logger.info('converting list into torch tensor')
    adjacency_matrix = tt.FloatTensor(adjacency_matrix)  # vocab_size, vocab_size
    initial_matrix = tt.FloatTensor(initial_matrix)  # vocab_size, 2

    # normalize
    logger.info('normalizing adjacency matrix to transition matrix')
    trans_matrix = normalize_trans_matrix(adjacency_matrix)

    return vocab_dict, initial_matrix, trans_matrix


def main():
    # load graph
    logger.info('loading graph')
    with open(args.graph_path, 'rb') as f_graph:
        graph = pickle.load(f_graph)
    vocab_dict, initial_matrix, trans_matrix = load_trans_matrix_from_graph(
        graph)

    information_matrix = stereotype_propagation(
        trans_matrix, initial_matrix, args.alpha)
    stereotype_lexicon = {}
    for word in vocab_dict:
        idx = vocab_dict[word]
        stereotype_lexicon[word] = float(information_matrix[idx])
    logger.info('saving lexicon')
    with open(args.stereotype_lexicon_path, 'w') as f_stereotype_lexicon:
        for word in stereotype_lexicon:
            word_info_str = f'{word}\t{stereotype_lexicon[word]:.3f}\n'
            f_stereotype_lexicon.write(word_info_str)


if __name__ == '__main__':
    main()



