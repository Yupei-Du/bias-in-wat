import math
from collections import Counter
import pickle
import copy

import pandas as pd
from tarjan import tarjan
from loguru import logger

from utils import parse_command_line_args


def read_csv_datafile(file_path):
    # read csv datafile as pandas data-frames
    df_data = pd.read_csv(file_path)
    return df_data


def is_legal(word):
    # whether word is NaN
    def is_empty(response):
        if isinstance(response, float):
            if math.isnan(response):
                return True
        else:
            return False

    unks = {'aeroplane', 'arse', 'ax', 'bandana', 'bannister', 'behaviour', 'bellybutton', 'centre',
            'cheque', 'chequered', 'chilli', 'colour', 'colours', 'corn-beef', 'cosy', 'doughnut',
            'extravert', 'favour', 'fibre', 'hanky', 'harbour', 'highschool', 'hippy', 'honour',
            'hotdog', 'humour', 'judgment', 'labour', 'light bulb', 'lollypop', 'neighbour',
            'neighbourhood', 'odour', 'oldfashioned', 'organisation', 'organise', 'paperclip',
            'parfum', 'phoney', 'plough', 'practise', 'programme', 'pyjamas',
            'racquet', 'realise', 'recieve', 'saviour', 'seperate', 'theatre', 'tresspass',
            'tyre', 'verandah', 'whisky', 'WIFI', 'yoghurt', 'smokey'}

    # if word is NaN or in unknown words
    if is_empty(word) or word in unks:
        return False
    else:
        return True


def unify_spellings(word):
    sub_dict = {'black out': 'blackout',
                'break up': 'breakup',
                'breast feeding': 'breastfeeding',
                'bubble gum': 'bubblegum',
                'cell phone': 'cellphone',
                'coca-cola': 'Coca Cola',
                'good looking': 'good-looking',
                'goodlooking': 'good-looking',
                'hard working': 'hardworking',
                'hard-working': 'hardworking',
                'lawn mower': 'lawnmower',
                'seat belt': 'seatbelt',
                'tinfoil': 'tin foil',
                'bluejay': 'blue jay',
                'bunk bed': 'bunkbed',
                'dingdong': 'ding dong',
                'dwarves': 'dwarfs',
                'Great Brittain': 'Great Britain',
                'lightyear': 'light year',
                'manmade': 'man made',
                'miniscule': 'minuscule',
                'pass over': 'passover'}

    if word in sub_dict:
        return sub_dict[word]
    else:
        return word


def preprocess(df_data):
    """
    Pre-process data. remove NaN and unknown words, unify spellings
    return dict-form processed data: {cue: Counter(responses)}
    """
    n_records = df_data.shape[0]

    processed_data = {}
    for participant_id, record in df_data.iterrows():
        if participant_id % 50000 == 0:
            logger.info(f'{participant_id} processed, {n_records} in total')

        cue = record['cue']
        # whether cue is legal
        if not is_legal(cue):
            continue
        cue = unify_spellings(cue)

        responses = []
        for response in (record['R1'], record['R2'], record['R3']):
            if is_legal(response):
                responses.append(unify_spellings(response))

        if cue not in processed_data:
            processed_data[cue] = responses
        else:
            processed_data[cue] += responses

    processed_data = {cue: Counter(processed_data[cue]) for cue in processed_data}

    return processed_data


def maximal_strongly_connected_component(processed_data):
    # find maximal strongly connected component of graph with tarjan algorithm
    graph = {cue: [response for response in processed_data[cue]] for cue in processed_data}
    scc_list = tarjan(graph)
    max_scc = sorted(scc_list, key=lambda x: len(x), reverse=True)[0]
    max_scc = set(max_scc)
    return max_scc


def build_graph(processed_data):
    male_words = ['father', 'son', 'he', 'grandfather', 'man',
                  'husband', 'brother', 'boy', 'uncle', 'gentleman']
    female_words = ['mother', 'daughter', 'she', 'grandmother', 'woman',
                    'wife', 'sister', 'girl', 'aunt', 'lady']

    logger.info('copying data')
    processed_data = copy.deepcopy(processed_data)

    # build vocab
    logger.info('building vocab dict')
    vocab = maximal_strongly_connected_component(processed_data)
    logger.info(f'vocab size: {len(vocab)}')

    # remove words not in the graph
    logger.info('removing words not in the graph')
    cues = list(processed_data.keys())
    for cue in cues:
        if cue not in vocab:
            del processed_data[cue]
        else:
            responses = list(processed_data[cue])
            for response in responses:
                if response not in vocab:
                    del processed_data[cue][response]

    # build vocab dict
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}

    # build initial matrix
    logger.info('building initial matrix')
    initial_matrix = [[0, 0] for _ in vocab_dict]
    for male_word in male_words:
        male_word_id = vocab_dict[male_word]
        initial_matrix[male_word_id] = [1, 0]
    for female_word in female_words:
        female_word_id = vocab_dict[female_word]
        initial_matrix[female_word_id] = [0, 1]

    # build adjacency matrix
    logger.info('building adjacency matrix')
    adjacency_matrix = [[0 for __ in range(len(vocab_dict))] for _ in range(len(vocab_dict))]

    for idx, cue in enumerate(processed_data):
        if idx % int(len(processed_data) / 10) == 0:
            logger.info(f'building process {idx} / {len(processed_data)}')
        cue_id = vocab_dict[cue]
        for response in processed_data[cue]:
            response_id = vocab_dict[response]
            # calculate weights
            adjacency_matrix[cue_id][response_id] += processed_data[cue][response]

    return vocab_dict, initial_matrix, adjacency_matrix


def main():
    logger.info('reading command line arguments')
    args = parse_command_line_args()

    # load processed data or re-process from raw
    if args.load_processed_data:
        logger.info('loading processed data')
        # load processed data from pkl file
        with open(args.processed_path, 'rb') as f_processed_pkl:
            processed_data = pickle.load(f_processed_pkl)
    else:
        logger.info('loading data from csv file')
        # read csv datafile as pandas data-frames
        df_data = read_csv_datafile(args.csv_path)
        # pre-process data-frame data to dict of counters
        logger.info('pre-processing data')
        processed_data = preprocess(df_data)
        # save processed data to pkl file
        with open(args.processed_path, 'wb') as f_processed_pkl:
            pickle.dump(processed_data, f_processed_pkl)

    # build graph
    logger.info('building graph')
    vocab_dict, initial_matrix, adjacency_matrix = build_graph(processed_data)
    graph = {
        'vocab_dict': vocab_dict,
        'initial_matrix': initial_matrix,
        'adjacency_matrix': adjacency_matrix,
    }

    # save
    logger.info('saving graph')
    with open(args.graph_path, 'wb') as f_graph:
        pickle.dump(graph, f_graph)


if __name__ == '__main__':
    main()
