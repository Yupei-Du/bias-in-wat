import argparse


def parse_command_line_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False

    parser = argparse.ArgumentParser()

    # system
    parser.add_argument('--gpu', type=int, default=-1)
    # path
    parser.add_argument('--csv_path', type=str, default='data/SWOW-EN.R100.csv')
    parser.add_argument('--processed_path', type=str, default='data/processed_data.pkl')
    parser.add_argument('--load_processed_data', type=str2bool, default=False)
    parser.add_argument('--graph_path', type=str, default='data/graph.pkl')
    parser.add_argument('--stereotype_lexicon_path', type=str,
                        default='data/stereotype_lexicon.txt')
    # hyper-parameters
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='alpha value in stereotype propagation')

    args = parser.parse_args()
    return args
