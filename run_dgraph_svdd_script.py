# %%
import dgl
from dgl.data.utils import load_graphs
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.model import HTGNN, NodePredictor
from utils.pytorchtools import EarlyStopping
from sklearn import metrics

dgl.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# %%
# device = torch.device('cuda:1')
device = torch.device('cpu')

# %%

train_feats, _ = load_graphs('./data/dgraph/train_feats.bin')
valid_feats, _ = load_graphs('./data/dgraph/valid_feats.bin')
test_feats, _ = load_graphs('./data/dgraph/test_feats.bin')

train_labels = torch.load("./data/dgraph/train_labels.pt")
valid_labels = torch.load("./data/dgraph/valid_labels.pt")
test_labels = torch.load("./data/dgraph/test_labels.pt")

# %%
train_feats[0], valid_feats[0]

# %%
train_feats[0].nodes['A'].data['feat'].shape

# %%
# train_labels[0], valid_labels[0]

# %%
time_window = 2

# %%
def valid_graph_feat(g_feat, time_window):
    all_etype_t = sorted(
        list(set([etype.split("_")[-1] for _, etype, _ in g_feat.canonical_etypes]))
    )

    if len(all_etype_t) >= time_window:
        return True
    else:
        return False

# %%
def write_to_file(value, fpath, name=None):
    with open(fpath, 'a') as fout:
        fout.write(f"{value}\n")

# %%
def evaluate(model, svdd, val_feats, val_labels, pred_node_type="ALL"):
    val_auc_list, val_ap_list = [], []

    model.eval()

    with torch.no_grad():
        for i, (G_feat, G_label) in enumerate(zip(val_feats, val_labels)):
            if not valid_graph_feat(G_feat, time_window):
                continue
            try:
                h = model[0](G_feat.to(device), pred_node_type)
                f_labels = []
                f_pred = []
                all_h = []
                for ntype in G_label.keys():
                    pred = svdd.compute_score(h[ntype]).view(-1, 1)
                    label = G_label[ntype].to(device).view(-1, 1)

                    label_mask = (label == 0) | (label == 1)

                    masked_label = label[label_mask]
                    masked_pred = pred[label_mask]

                    f_labels.append(masked_label)
                    f_pred.append(masked_pred)

                f_labels = torch.cat(f_labels)
                f_pred = torch.cat(f_pred)

            except Exception as e:
                print(f"failed val index: {i}")
                raise Exception(e)

            if f_labels.unique().shape[0] >= 2:
                # AUC
                fpr, tpr, thresholds = metrics.roc_curve(
                    f_labels.numpy(), f_pred.numpy()
                )
                auc = metrics.auc(fpr, tpr)

                # AP
                precision, recall, thresholds = metrics.precision_recall_curve(
                    f_labels.numpy(), f_pred.numpy()
                )
                ap = metrics.auc(recall, precision)

                val_auc_list.append(auc)
                val_ap_list.append(ap)

        auc = sum(val_auc_list) / len(val_auc_list)
        ap = sum(val_ap_list) / len(val_ap_list)

        print(f"\tEval AUC/AP: {auc} / {ap}")

    return auc, ap

# %%


# %%

graph_atom = train_feats[10]
mae_list, rmse_list = [], []
model_out_path = 'checkpoint'


# %%


# %%
htgnn = HTGNN(graph=graph_atom, n_inp=16, n_hid=8, n_layers=2, n_heads=1, time_window=time_window, norm=False, device=device).to(device)
predictor = NodePredictor(n_inp=8, n_classes=1).to(device)
model = nn.Sequential(htgnn, predictor).to(device)

# %%
early_stopping = EarlyStopping(patience=10, verbose=True, path=f'{model_out_path}/checkpoint_HTGNN.pt')
optim = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

# train_mae_list, train_rmse_list = [], []
train_svdd_list = []
idx = np.random.permutation(len(train_feats))

# %%
class SVDDLoss:
    def __init__(self) -> None:
        self.center = None
        self.l2_lambda = 0.001
        self.save_path = "./results/dgraph_svdd"

    def set_svdd_center(self, center):
        self.center = center

    def load_svdd_center(self, fpath):
        raise Exception("Not Implemented!")

    def save_svdd_center(self):
        torch.save(self.center, f"{self.save_path}/SVDD_Center.pt")

    def compute_svdd_loss(self, model, node_embeddings):
        if self.center is None:
            with torch.no_grad():
                center = torch.mean(node_embeddings, 0)
                self.set_svdd_center(center)
                self.save_svdd_center()

        dist = torch.sum(torch.square(node_embeddings - self.center), 1)
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()) / 2
        _loss = torch.mean(dist)

        svdd_loss = _loss + self.l2_lambda * l2_norm
        return svdd_loss
    
    def compute_score(self, node_embeddings):
        dist = torch.mean(torch.square(node_embeddings - self.center), 1)
        return dist

# %%
pred_node_type = "ALL"
save_path = "./results/dgraph_svdd"

svdd = SVDDLoss()

for epoch in range(200):
    model.train()

    print(f"============ Epoch {epoch} ============")
    for i in tqdm(idx):
        G_feat = train_feats[i].to(device)
        G_label = train_labels[i]

        # check if graph contains more than 2 windows
        if not valid_graph_feat(G_feat, time_window):
            continue

        h = model[0](G_feat, pred_node_type)

        all_h = []
        f_labels = []
        for ntype in G_label.keys():
            label = G_label[ntype].to(device).view(-1, 1)

            label_mask = (label == 0) | (label == 1)

            masked_label = label[label_mask]

            f_labels.append(masked_label)
            all_h.append(h[ntype])

        f_labels = torch.cat(f_labels)
        all_h = torch.cat(all_h, 0)

        loss = svdd.compute_svdd_loss(model[0], all_h)
        train_svdd_list.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    epoch_loss = sum(train_svdd_list) / len(train_svdd_list)
    print(f"Epoch SVDD Loss: {epoch_loss}")

    write_to_file(epoch_loss, f"{save_path}/train_svdd_loss.txt")

    if epoch % 2 == 0:
        auc, ap = evaluate(model, svdd, valid_feats, valid_labels)
        write_to_file(auc, f"{save_path}/eval_auc.txt")
        write_to_file(ap, f"{save_path}/eval_ap.txt")
        early_stopping(loss, model)

# %%



