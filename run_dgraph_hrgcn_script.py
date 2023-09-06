# %%
import dgl
from dgl.data.utils import load_graphs
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.HRTGCN import HRTGCN, NodePredictor
from utils.pytorchtools import EarlyStopping
from sklearn import metrics

dgl.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# %%
# device = torch.device('cuda:1')
device = torch.device('cuda')

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
def evaluate(model, val_feats, val_labels, pred_node_type="ALL"):
    val_mae_list, val_rmse_list = [], []
    val_auc_list, val_ap_list = [], []

    model.eval()

    with torch.no_grad():
        for i, (G_feat, G_label) in enumerate(zip(val_feats, val_labels)):
            if not valid_graph_feat(G_feat.to(device), time_window):
                continue
            try:
                h = model[0](G_feat.to(device), pred_node_type)
                f_labels = []
                f_pred = []
                for ntype in G_label.keys():
                    pred = model[1](h[ntype])
                    label = G_label[ntype].to(device).view(-1, 1)

                    label_mask = (label == 0) | (label == 1)

                    masked_label = label[label_mask]
                    masked_pred = pred[label_mask]

                    f_labels.append(masked_label)
                    f_pred.append(masked_pred)

                f_labels = torch.cat(f_labels)
                f_pred = torch.cat(f_pred)

                loss = F.l1_loss(f_pred, f_labels)
                rmse = torch.sqrt(F.mse_loss(f_pred, f_labels))
            except Exception as e:
                print(f"failed val index: {i}")
                raise Exception(e)

            val_mae_list.append(loss.item())
            val_rmse_list.append(rmse.item())

            if f_labels.unique().shape[0] >= 2:
                # AUC
                fpr, tpr, thresholds = metrics.roc_curve(
                    f_labels.cpu().numpy(), f_pred.cpu().numpy()
                )
                auc = metrics.auc(fpr, tpr)

                # AP
                precision, recall, thresholds = metrics.precision_recall_curve(
                    f_labels.cpu().numpy(), f_pred.cpu().numpy()
                )
                ap = metrics.auc(recall, precision)

                val_auc_list.append(auc)
                val_ap_list.append(ap)

        loss = sum(val_mae_list) / len(val_mae_list)
        rmse = sum(val_rmse_list) / len(val_rmse_list)

        auc = sum(val_auc_list) / len(val_auc_list)
        ap = sum(val_ap_list) / len(val_ap_list)

        print(f"\tEval MAE/RMSE: {loss} / {rmse}")
        print(f"\tEval AUC/AP: {auc} / {ap}")

    return loss, rmse, auc, ap

# %%


# %%

graph_atom = train_feats[10]
mae_list, rmse_list = [], []
model_out_path = 'checkpoint'


# %%
htgnn = HRTGCN(graph=graph_atom, n_inp=16, n_hid=8, n_layers=2, n_heads=1, time_window=time_window, norm=False, device=device).to(device)
predictor = NodePredictor(n_inp=8, n_classes=1).to(device)
model = nn.Sequential(htgnn, predictor).to(device)

# %%
early_stopping = EarlyStopping(patience=10, verbose=True, path=f'{model_out_path}/checkpoint_HTGNN.pt')
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

train_mae_list, train_rmse_list = [], []
idx = np.random.permutation(len(train_feats))

# %%
output_dir = "./results/dgraph_hrgcn"

# %%
pred_node_type = "ALL"

for epoch in range(200):
    model.train()

    print(f"============ Epoch {epoch} ============")
    for i in tqdm(idx):
        G_feat = train_feats[i]
        G_label = train_labels[i]

        # check if graph contains more than 2 windows
        if not valid_graph_feat(G_feat.to(device), time_window):
            continue

        h = model[0](G_feat.to(device), pred_node_type)

        f_labels = []
        f_pred = []
        for ntype in G_label.keys():
            pred = model[1](h[ntype])
            label = G_label[ntype].to(device).view(-1, 1)

            label_mask = (label == 0) | (label == 1)

            masked_label = label[label_mask]
            masked_pred = pred[label_mask]

            f_labels.append(masked_label)
            f_pred.append(masked_pred)

        f_labels = torch.cat(f_labels)
        f_pred = torch.cat(f_pred)

        loss = F.l1_loss(f_pred, f_labels)
        rmse = torch.sqrt(F.mse_loss(f_pred, f_labels))

        train_mae_list.append(loss.item())
        train_rmse_list.append(rmse.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

    epoch_mae = sum(train_mae_list) / len(train_mae_list)
    epoch_rmse = sum(train_rmse_list) / len(train_rmse_list)
    print(f"Epoch MAE/RMSE: {epoch_mae} / {epoch_rmse}")

    write_to_file(epoch_mae, f"{output_dir}/train_mae.txt")
    write_to_file(epoch_rmse, f"{output_dir}/train_rmse.txt")

    if epoch % 2 == 0:
        loss, rmse, auc, ap = evaluate(model, valid_feats, valid_labels)
        write_to_file(loss, f"{output_dir}/eval_mae.txt")
        write_to_file(rmse, f"{output_dir}/eval_rmse.txt")
        write_to_file(auc, f"{output_dir}/eval_auc.txt")
        write_to_file(ap, f"{output_dir}/eval_ap.txt")
        early_stopping(loss, model)

# %%



