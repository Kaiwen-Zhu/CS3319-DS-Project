import os
import torch
import dgl
import numpy as np
import logging
import gc
import time
import pickle as pkl
from tqdm import tqdm

from config import make_argparser
from build_graph import build_graph
from model import GNN
from evaluate import evaluate_on_testset, submit


args = make_argparser()

enhance_flag = '_enhance' if args.enhance else ''
save_path = os.path.join(args.save_root, f"{int(args.train_frac*10)}_{str(round(time.time()))[-6:]}{enhance_flag}")
os.mkdir(save_path)

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("Arguments:")
for k, v in vars(args).items():
    logger.info(f'  {k}: {v}')


os.environ['DGLBACKEND'] = 'pytorch'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from_path = os.path.join(args.save_root, args.from_dir) if args.enhance else None
G, node_features, rel_list, pred_weight, paper_feature, edges, refs_to_pred, n_author, n_paper = build_graph(
    args.data_root, args.train_frac, save_path, device, logger, 
    init_author_path=args.init_author_path, add_write=args.add_write,
    enhance=args.enhance, from_path=from_path, enhance_frac=args.enhance_frac, load_pair=args.load_pair)

sampler = dgl.dataloading.NeighborSampler([5, 5, 5, 5])
sampler = dgl.dataloading.as_edge_prediction_sampler(
    sampler, negative_sampler=dgl.dataloading.negative_sampler.GlobalUniform(7))
train_eid_dict = {etype: G.edges(etype=etype, form='eid') for etype in G.etypes}
dataloader = dgl.dataloading.DataLoader(
    G, train_eid_dict, sampler,
    device=device, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

# Train
model = GNN(paper_feature.shape[1], *args.dims, rel_list, device, pred_weight).to(device)
opt = torch.optim.Adam(model.parameters())
n_batches = len(dataloader)
losses = []
with tqdm(range(args.epochs), desc="Training GNN", unit='epoch', ncols=100) as pbar:
    for epoch in pbar:
        epoch_loss = 0
        for input_nodes, positive_graph, negative_graph, blocks in dataloader:
            input_features = blocks[0].srcdata['features']
            pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
            loss = model.loss_fn(pos_score, neg_score, rel_list[0])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        gc.collect()
        torch.cuda.empty_cache()
        epoch_loss /= n_batches
        losses.append(epoch_loss)
        pbar.set_postfix({"Epoch loss": f"{epoch_loss:.4f}"})
        logger.debug(f"Epoch {epoch}: loss = {epoch_loss:.4f}")
with open(os.path.join(save_path, 'losses.pkl'), 'wb') as f:
    pkl.dump(losses, f)

# Compute node embeddings
with torch.no_grad():
    node_embeddings = model.rgcn([G, G, G, G], node_features)
author_emb = np.array(node_embeddings['author'].detach().cpu())
paper_emb = np.array(node_embeddings['paper'].detach().cpu())
emb = {'author': author_emb, 'paper': paper_emb}
with open(os.path.join(save_path, 'node_embeddings.pkl'), 'wb') as f:
    pkl.dump(emb, f)
  
# Evaluate
test_refs, test_cites, test_cos = edges['test_refs'], edges['test_cites'], edges['test_cos']
threshold = evaluate_on_testset(test_refs, test_cites, test_cos, author_emb, paper_emb, logger)

submit(refs_to_pred, author_emb, paper_emb, threshold, save_path)

logger.info(f"Results saved to {save_path}.")
