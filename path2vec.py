from dgl.nn.pytorch import MetaPath2Vec
import torch
from torch.utils.data import DataLoader
from typing import List


def get_author_init(G, device, rel_list: List[str], exist_path: str = None):
    """Gets the initial embedding for the author nodes using the MetaPath2Vec algorithm.
    If the exist_path is not None, then the embedding is loaded from the path."""    

    if exist_path is not None:
        return torch.load(exist_path)

    p2v_model = MetaPath2Vec(G, [rel_list[0][1], rel_list[1][1], rel_list[3][1], rel_list[2][1]], window_size=2, emb_dim=512).to(device)
    dataloader = DataLoader(torch.arange(G.num_nodes('author')), batch_size=1024,
                            shuffle=True, collate_fn=p2v_model.sample)
    opt = torch.optim.SparseAdam(p2v_model.parameters(), lr=0.01)

    for epoch in range(1000):
        epoch_loss = 0
        for (pos_u, pos_v, neg_v) in dataloader:
            pos_u = pos_u.to(device)
            pos_v = pos_v.to(device)
            neg_v = neg_v.to(device)

            loss = p2v_model(pos_u, pos_v, neg_v)
            epoch_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: loss = {epoch_loss:.3f}")
    opt.zero_grad()
    
    author_nids = torch.LongTensor(p2v_model.local_to_global_nid['author']).to(device)
    with torch.no_grad():
        author_init = p2v_model.node_embed(author_nids)
    torch.save(author_init, 'author_init.bin')
    return author_init
