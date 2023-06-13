from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle as pkl
import argparse
from load_data import read_txt


def cos_sim(a, b):
    return np.sum(a*b, axis=1) / (np.linalg.norm(a,axis=1) * np.linalg.norm(b,axis=1))


def evaluate(test_df, emb1, emb2):
    test_arr = np.array(test_df.values)
    res = cos_sim(emb1[test_arr[:,0]], emb2[test_arr[:,1]])
    lbl_true = test_df.label.to_numpy()
    precision, recall, threshold = precision_recall_curve(lbl_true, np.array(res))
    # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    # disp.plot()
    # plt.show()
    f1 = np.array([2*precision[i]*recall[i] / (precision[i]+recall[i]) 
                    if precision[i]+recall[i] else 0
                    for i in range(len(precision))])
    ind = f1.argmax()
    prec, rec, thresh = precision[ind], recall[ind], threshold[ind]
    return f1.max(), prec, rec, thresh


def evaluate_on_testset(test_refs, test_cites, test_cos, author_emb, paper_emb, logger):
    ref_f1_max, ref_prec, ref_rec, threshold = evaluate(test_refs, author_emb, paper_emb)
    cite_f1_max, cite_prec, cite_rec, _ = evaluate(test_cites, paper_emb, paper_emb)
    co_f1_max, co_prec, co_rec, _ = evaluate(test_cos, author_emb, author_emb)
    logger.info(f"Max f1 score of predicting reference edges: {ref_f1_max:.4f} (precision = {ref_prec:.4f}, recall = {ref_rec:.4f}), \
threshold = {threshold:.4f}")
    logger.info(f"Max f1 score of predicting citation edges: {cite_f1_max:.4f} (precision = {cite_prec:.4f}, recall = {cite_rec:.4f})")
    logger.info(f"Max f1 score of predicting coauthor edges: {co_f1_max:.4f} (precision = {co_prec:.4f}, recall = {co_rec:.4f})")
    return threshold


def submit(refs_to_pred, author_emb, paper_emb, threshold, save_path):
    test_arr = np.array(refs_to_pred)
    res = cos_sim(author_emb[test_arr[:, 0]], paper_emb[test_arr[:, 1]])
    res[res >= threshold] = 1
    res[res < threshold] = 0
    data = []
    for index, p in enumerate(res):
        data.append([index, str(int(p))])
    df = pd.DataFrame(data, columns=["Index", "Predicted"], dtype=object)
    df.to_csv(os.path.join(save_path, 'Submission.csv'), index=False)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--test_path', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'test.txt'))
    arg_parser.add_argument('--source_dir', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'best'))
    arg_parser.add_argument('--save_path', type=str, default=
        os.path.abspath(os.path.dirname(__file__)))
    args = arg_parser.parse_args()

    refs_to_pred = read_txt(args.test_path)
    with open(os.path.join(args.source_dir, 'threshold.txt'), 'r') as f:
        threshold = float(f.read())
    with open(os.path.join(args.source_dir, 'node_embeddings.pkl'), 'rb') as f:
        embeddings = pkl.load(f)
    submit(refs_to_pred, embeddings['author'], embeddings['paper'], threshold, args.save_path)
