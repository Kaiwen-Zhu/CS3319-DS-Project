import os
import pickle as pkl
import numpy as np
import pandas as pd
from typing import List


def read_txt(filepath: str) -> List[List[int]]:
    """Reads list of edges."""  

    res_list = list()
    with open(filepath, "r") as f:
        line_list = f.readlines()
    for line in line_list:
        res_list.append(list(map(int, line.strip().split(' '))))      
    return res_list


def load_data(data_root: str):
    """Loads edges and paper features."""    

    cite_file = "paper_file_ann.txt"
    train_ref_file = "bipartite_train_ann.txt"
    test_ref_file = "bipartite_test_ann.txt"
    coauthor_file = "author_file_ann.txt"
    feature_file = "feature.pkl"

    citation = read_txt(os.path.join(data_root, cite_file))
    existing_refs = read_txt(os.path.join(data_root, train_ref_file))
    refs_to_pred = read_txt(os.path.join(data_root, test_ref_file))
    coauthor = read_txt(os.path.join(data_root, coauthor_file))

    feature_file = os.path.join(data_root, feature_file)
    with open(feature_file, 'rb') as f:
        paper_feature = pkl.load(f)   

    return citation, existing_refs, refs_to_pred, coauthor, paper_feature


def split_train_test(edges: List[List[int]], train_frac: float, n_source: int, n_target: int):
    """Splits edges into train and test sets."""

    df = pd.DataFrame(edges, columns=['source', 'target'])
    train_edges = df.sample(frac=train_frac, random_state=0, axis=0)

    test_true_edges = df.copy()[~df.index.isin(train_edges.index)]
    test_true_edges.loc[:, 'label'] = 1

    n_test_edges = len(test_true_edges)
    true_set = set(map(tuple, edges))
    cnt = 0
    test_false_edges = []
    while cnt < n_test_edges:
        s = np.random.randint(n_source)
        t = np.random.randint(n_target)
        if (s, t) not in true_set:
            test_false_edges.append([s,t])
            cnt += 1
    test_false_edges = pd.DataFrame(test_false_edges, columns=['source', 'target'])
    test_false_edges.loc[:, 'label'] = 0

    test_edges = pd.concat([test_true_edges, test_false_edges])

    return train_edges, test_edges


def get_train_test_data(data_root: str, train_frac: float, logger, save_path: str = None):
    """Loads and splits data into train and test sets."""    

    citation, existing_refs, refs_to_pred, coauthor, paper_feature = load_data(data_root)

    authors = set(edge[0] for edge in existing_refs) | set(edge[0] for edge in refs_to_pred) \
            | set(edge[0] for edge in coauthor) | set(edge[1] for edge in coauthor)
    papers = set(edge[1] for edge in existing_refs) | set(edge[1] for edge in refs_to_pred) \
            | set(edge[0] for edge in citation) | set(edge[1] for edge in citation)

    n_author, n_paper = len(authors), len(papers)
    logger.info(f"#authors: {len(authors)}, #papers: {len(papers)}")

    train_refs, test_refs = split_train_test(existing_refs, train_frac, n_author, n_paper)
    train_cites, test_cites = split_train_test(citation, train_frac, n_paper, n_paper)
    train_cos, test_cos = split_train_test(coauthor, train_frac, n_author, n_author)

    logger.info(f"#(author, paper) edge in train set: {len(train_refs)}, in test set: {len(test_refs)}")
    logger.info(f"#(paper, paper) edge in train set: {len(train_cites)}, in test set: {len(test_cites)}")
    logger.info(f"#(author, author) edge in train set: {len(train_cos)}, in test set: {len(test_cos)}")

    node_tmp = pd.concat([train_refs['source'], train_cos['source'], train_cos['target']])
    node_authors = pd.DataFrame(index=pd.unique(node_tmp))

    node_tmp = pd.concat([train_cites.loc[:, 'source'], train_cites.loc[:, 'target'], train_refs.loc[:, 'target']])
    node_papers = pd.DataFrame(index=pd.unique(node_tmp))

    logger.info(f"#authors in train set: {len(node_authors)}")
    logger.info(f"#papers in train set: {len(node_papers)}")

    edges = {'train_refs': train_refs,
        'test_refs': test_refs,
        'train_cites': train_cites,
        'test_cites': test_cites,
        'train_cos': train_cos,
        'test_cos': test_cos}

    if save_path is not None:
        with open(os.path.join(save_path, 'edges.pkl'), 'wb') as f:
            pkl.dump(edges, f)

    return edges, refs_to_pred, paper_feature, n_author, n_paper
