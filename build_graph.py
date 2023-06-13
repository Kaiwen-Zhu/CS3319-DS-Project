import torch
import dgl
import pickle as pkl
import os
from load_data import get_train_test_data
from path2vec import get_author_init
from enhance_edge import get_enhanced_edge
from find_writing import find_writing


def build_graph(data_root, train_frac, save_path, device, logger,
                init_author_path=None, add_write=False, enhance=False,
                from_path=None, enhance_frac=None, load_pair=False):
    edges, refs_to_pred, paper_feature, n_author, n_paper = get_train_test_data(
                    data_root, train_frac, logger, save_path)

    if not enhance:
        train_refs, train_cites, train_cos = edges['train_refs'], edges['train_cites'], edges['train_cos']
        train_ref_tensor = torch.from_numpy(train_refs.values).to(device)
        cite_tensor = torch.from_numpy(train_cites.values).to(device)
        coauthor_tensor = torch.from_numpy(train_cos.values).to(device)

        if not add_write:
            rel_list = [('author', 'ref', 'paper'), ('paper', 'cite', 'paper'), 
                        ('author', 'coauthor', 'author'), ('paper', 'beref', 'author')]
            graph_data = {
                rel_list[0]: (train_ref_tensor[:, 0], train_ref_tensor[:, 1]),
                rel_list[1]: (torch.cat([cite_tensor[:, 0], cite_tensor[:, 1], torch.arange(n_paper, device=device)]),
                            torch.cat([cite_tensor[:, 1], cite_tensor[:, 0], torch.arange(n_paper, device=device)])),
                rel_list[2]: (torch.cat([coauthor_tensor[:, 0], coauthor_tensor[:, 1], torch.arange(n_author, device=device)]), 
                            torch.cat([coauthor_tensor[:, 1], coauthor_tensor[:, 0], torch.arange(n_author, device=device)])),
                rel_list[3]: (train_ref_tensor[:, 1], train_ref_tensor[:, 0])
            }
            pred_weight = None
        else:
            likely_write_edges, likelihood = find_writing(
                edges['train_refs'], edges['train_cites'], n_author, n_paper, logger)
            write_tensor = torch.from_numpy(likely_write_edges.values).to(device)
            rel_list = [('author', 'ref', 'paper'), ('paper', 'cite', 'paper'), 
                        ('author', 'coauthor', 'author'), ('paper', 'beref', 'author'),
                        ('author', 'write', 'paper')]
            graph_data = {
                rel_list[0]: (train_ref_tensor[:, 0], train_ref_tensor[:, 1]),
                rel_list[1]: (torch.cat([cite_tensor[:, 0], cite_tensor[:, 1]]), torch.cat([cite_tensor[:, 1], cite_tensor[:, 0]])),
                rel_list[2]: (torch.cat([coauthor_tensor[:, 0], coauthor_tensor[:, 1]]), torch.cat([coauthor_tensor[:, 1], coauthor_tensor[:, 0]])),
                rel_list[3]: (train_ref_tensor[:, 1], train_ref_tensor[:, 0]),
                rel_list[4]: (write_tensor[:, 0], write_tensor[:, 1])
            }
            pred_weight = {'write': torch.tensor(likelihood).to(device)}

    else:
        with open(os.path.join(from_path, "edges.pkl"), "rb") as f:
            edges = pkl.load(f)
        train_refs, train_cites, train_cos = edges['train_refs'], edges['train_cites'], edges['train_cos']
        train_ref_tensor = torch.from_numpy(train_refs.values).to(device)
        cite_tensor = torch.from_numpy(train_cites.values).to(device)
        coauthor_tensor = torch.from_numpy(train_cos.values).to(device)
        with open(os.path.join(from_path, "node_embeddings.pkl"), "rb") as f:
            embeddings = pkl.load(f)
        
        pair_path = {"load": load_pair, "path": f"{int(10*train_frac)}_pairs.pkl"}
        aa_pred_tensor, pp_pred_tensor, ap_pred_tensor, pred_weight = get_enhanced_edge(
            train_refs, train_cites, train_cos, embeddings['author'], embeddings['paper'], 
            enhance_frac, device, logger, pair_path)
        rel_list = [('author', 'ref', 'paper'), ('paper', 'cite', 'paper'), 
                    ('author', 'coauthor', 'author'), ('paper', 'beref', 'author'),
                    ('author', 'aa_pred', 'author'), ('paper', 'pp_pred', 'paper'), 
                    ('author', 'ap_pred', 'paper')]
        graph_data = {
            rel_list[0]: (train_ref_tensor[:, 0], train_ref_tensor[:, 1]),
            rel_list[1]: (torch.cat([cite_tensor[:, 0], cite_tensor[:, 1], torch.arange(n_paper, device=device)]),
                        torch.cat([cite_tensor[:, 1], cite_tensor[:, 0], torch.arange(n_paper, device=device)])),
            rel_list[2]: (torch.cat([coauthor_tensor[:, 0], coauthor_tensor[:, 1], torch.arange(n_author, device=device)]), 
                        torch.cat([coauthor_tensor[:, 1], coauthor_tensor[:, 0], torch.arange(n_author, device=device)])),
            rel_list[3]: (train_ref_tensor[:, 1], train_ref_tensor[:, 0]),
            rel_list[4]: (aa_pred_tensor[:, 0], aa_pred_tensor[:, 1]),
            rel_list[5]: (pp_pred_tensor[:, 0], pp_pred_tensor[:, 1]),
            rel_list[6]: (ap_pred_tensor[:, 0], ap_pred_tensor[:, 1])
        }
    
    hetero_graph = dgl.heterograph(graph_data)

    author_init = get_author_init(hetero_graph, device, rel_list, exist_path=init_author_path)
    node_features = {'author': author_init, 'paper': paper_feature.to(device)}
    hetero_graph.ndata['features'] = node_features
    hetero_graph = hetero_graph.to(device)

    logger.info(hetero_graph)

    return hetero_graph, node_features, rel_list, pred_weight, paper_feature, edges, refs_to_pred, n_author, n_paper
