import torch
import torch.nn.functional as F
import numpy as np
from sys import stdout
from tqdm import tqdm
from typing import List, Tuple
import gc
import pickle as pkl


def topK(emb1: np.ndarray, emb2: np.ndarray, thresh: float, 
            exist_set: set, same: bool = False):
    norm1 = np.linalg.norm(emb1, axis=1)
    norm2 = np.linalg.norm(emb2, axis=1)
    norms = norm1.reshape(-1,1) @ norm2.reshape(1,-1)
    sims = (emb1 @ emb2.T / norms).clip(max=1)
    if same:
        sims[np.tril_indices(sims.shape[0])] = -1
    indices = np.where(sims>thresh)
    pairs = list(filter(lambda t: t[:2] not in exist_set, zip(*indices, sims[indices])))
    pairs.sort(key=lambda t: t[-1], reverse=True)
    return pairs


def LSH_topK(vecs: np.ndarray, b: int, r: int, exist_set: set, thresh: float, device: str) -> List[Tuple[int, int, float]]:
    """Finds the most similar pairs using LSH for cosine similarity.

    Args:
        vecs (np.ndarray): Vectors to find similar pairs for.
        b (int): Number of bands.
        r (int): Number of rows in each band.
        
    Returns:
        List[Tuple[int, int, float]]: Tuples containing the indices of the similar pairs and the cosine similarity.
    """        

    n = b * r
    N, D = vecs.shape

    # Generate random vectors
    rand_vecs = np.random.randn(n, D)

    # Compute the hash values for all vectors
    signatures = (rand_vecs @ vecs.T) > 0

    # Find candidate pairs
    candidates = set()
    for i in tqdm(range(b), desc="\tLooking for candidate pairs", unit='band', ncols=80):
        buckets = {}
        band = signatures[i*r:(i+1)*r, :]

        # Find duplicate columns in band
        for j in range(N):
            col = tuple(band[:, j])
            if col in buckets:
                buckets[col].append(j)
            else:
                buckets[col] = [j]

        # Add all pairs in the same bucket to candidates
        for bucket in buckets.values():
            for j in range(len(bucket)):
                for k in range(j+1, len(bucket)):
                    candidates.add((bucket[j], bucket[k]))
    candidates = list(candidates)

    # Compute the actual cosine similarity for each candidate pair
    stdout.flush()
    print("\t#candidates:", len(candidates))
    batch_size = 1 << 20
    pairs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(candidates), batch_size), desc="\tChecking candidate pairs", ncols=80):
            cand_idx0 = np.array([t[0] for t in candidates[i:i+batch_size]])
            cand_idx1 = np.array([t[1] for t in candidates[i:i+batch_size]])
            tensor0 = torch.from_numpy(vecs[cand_idx0]).to(device)
            tensor1 = torch.from_numpy(vecs[cand_idx1]).to(device)
            cand_sim = F.cosine_similarity(tensor0, tensor1, dim=1).cpu().numpy()
            pairs.extend(list(filter(
                lambda t: t[2] > thresh and t[:2] not in exist_set, 
                zip(cand_idx0, cand_idx1, cand_sim.clip(max=1)))))
            gc.collect()
            torch.cuda.empty_cache()
    
    # Sort the candidate pairs by cosine similarity
    print("\tSorting...")
    stdout.flush()
    pairs.sort(key=lambda x: x[-1], reverse=True)
    return pairs


def get_enhanced_edge(train_refs, train_cites, train_cos, author_emb, paper_emb, 
                        enhance_frac, device, logger, pair_path):
    citation_set = set(map(lambda t: (min(t), max(t)), train_cites.values))
    ref_set = set(map(tuple, train_refs.values))
    coauthor_set = set(map(lambda t: (min(t), max(t)), train_cos.values))

    if not pair_path['load']:
        print("Looking for author-author edges...", end=' ')
        stdout.flush()
        aa_pairs = topK(author_emb, author_emb, thresh=0.9, exist_set=coauthor_set, same=True)
        print("Done")
        print("Looking for author-paper edges...", end=' ')
        stdout.flush()
        ap_pairs = topK(author_emb, paper_emb, thresh=0, exist_set=ref_set, same=False)
        print("Done")
        print("Looking for paper-paper edges...")
        stdout.flush()
        pp_pairs = LSH_topK(paper_emb, b=1, r=15, exist_set=citation_set, thresh=0.9, device=device)
        print("Done")
        logger.info(f"#filtered author-author edges: {len(aa_pairs)}")
        logger.info(f"#filtered author-paper edges: {len(ap_pairs)}")
        logger.info(f"#filtered paper-paper edges: {len(pp_pairs)}")
        pairs = {'aa_pairs': aa_pairs,
            'ap_pairs': ap_pairs,
            'pp_pairs': pp_pairs}

        with open(pair_path['path'], 'wb') as f:
            pkl.dump(pairs, f)
    else:
        with open(pair_path['path'], 'rb') as f:
            pairs = pkl.load(f)
        aa_pairs, ap_pairs, pp_pairs = pairs['aa_pairs'], pairs['ap_pairs'], pairs['pp_pairs']

    n_aa = len(coauthor_set)
    n_ap = len(ref_set)
    n_pp = len(citation_set)
    K = (int(n_aa*enhance_frac), int(n_ap*enhance_frac), int(n_pp*enhance_frac))
    logger.info(f"#desired predicted author-author edges: {K[0]}")
    logger.info(f"#desired predicted author-paper edges: {K[1]}")
    logger.info(f"#desired predicted paper-paper edges: {K[2]}")
    aa_pairs = aa_pairs[:K[0]]
    ap_pairs = ap_pairs[:K[1]]
    pp_pairs = pp_pairs[:K[2]]
        

    def get_edge_and_weight(pairs):
        return [t[:2] for t in pairs], [t[2] for t in pairs]

    aa_pred, aa_weight = get_edge_and_weight(aa_pairs)
    ap_pred, ap_weight = get_edge_and_weight(ap_pairs)
    ap_weight = [(w+1)/2 for w in ap_weight]
    pp_pred, pp_weight = get_edge_and_weight(pp_pairs)

    aa_pred_tensor = torch.tensor(aa_pred).to(device)
    pp_pred_tensor = torch.tensor(pp_pred).to(device)
    ap_pred_tensor = torch.tensor(ap_pred).to(device)
    pred_weight = {
        'aa_pred': torch.tensor(aa_weight).to(device),
        'pp_pred': torch.tensor(pp_weight).to(device), 
        'ap_pred': torch.tensor(ap_weight).to(device)
    }

    logger.info(f"#predicted author-author edges: {len(aa_pred)}, min similarity is {aa_weight[-1]:.4f}")
    logger.info(f"#predicted author-paper edges: {len(ap_pred)}, min similarity is {ap_weight[-1]:.4f}")
    logger.info(f"#predicted paper-paper edges: {len(pp_pred)}, min similarity is {pp_weight[-1]:.4f}")

    return aa_pred_tensor, pp_pred_tensor, ap_pred_tensor, pred_weight
