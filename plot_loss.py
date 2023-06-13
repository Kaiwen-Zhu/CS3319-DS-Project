import pickle as pkl
import matplotlib.pyplot as plt
import os


with open(os.path.join("checkpoints", '9', 'losses.pkl'), 'rb') as f:
    ori_loss = pkl.load(f)

with open(os.path.join("checkpoints", '9_477494', 'losses.pkl'), 'rb') as f:
    simple_loss = pkl.load(f)

with open(os.path.join("checkpoints", '9_4_374437_enhance', 'losses.pkl'), 'rb') as f:
    further_loss = pkl.load(f)

plt.plot(ori_loss, alpha=0.5, label='without edge enhancement')
plt.plot(simple_loss, alpha=0.5, label='simple edge enhancement')
plt.plot(further_loss, alpha=0.5, label='further edge enhancement')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png", dpi=300)
