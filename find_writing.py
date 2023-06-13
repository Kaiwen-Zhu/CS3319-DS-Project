from tqdm import tqdm
import pandas as pd


def find_writing(ref_edges, cite_edges, n_author, n_paper, logger):
    ref_list = [0] * n_author
    for author in range(n_author):
        ref_list[author] = set(ref_edges[ref_edges['source']==author]['target'].values)

    likely_write_edges = []
    likelihood = []
    for paper in tqdm(range(n_paper), desc='Looking for possible writing relationships', unit='paper'):
        cite_set = set(cite_edges[cite_edges['source']==paper]['target'].values)
        n_cited = len(cite_set)
        if n_cited:
            for author in range(n_author):
                inter = cite_set & ref_list[author]
                prop = len(inter) / n_cited
                if prop > 0.7:
                    likely_write_edges.append((author, paper))
                    likelihood.append(prop)

    likely_write_edges = pd.DataFrame(likely_write_edges, columns=['source', 'target'])
    logger.info(f"#predicted writing edges: {len(likely_write_edges)}")

    return likely_write_edges, likelihood
