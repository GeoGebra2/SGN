import os
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from model import SGN
from data import NTUDataLoaders

args = type('A', (), {})()
args.dataset = 'NTU_ID'
args.case = 0
args.seg = 20
args.use_position_stream = 0
args.use_velocity_stream = 1
args.use_acceleration_stream = 1
args.use_angular_velocity_stream = 1
args.disentangle = 1
args.grl_lambda = 1.0

num_classes = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SGN(num_classes, args.dataset, args.seg, args).to(device).eval()
ckpt = 'results/NTU_ID/SGN/0_best.pth'
sd = torch.load(ckpt)
model.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd)

def get_loader(dataset):
    loaders = NTUDataLoaders(dataset=dataset, case=args.case, seg=args.seg, args=args)
    return loaders.get_val_loader(batch_size=64, num_workers=16)

buf = {}
def hook(mod, inp):
    buf['x'] = inp[0].detach().cpu()
model.fc.register_forward_pre_hook(lambda m, inp: hook(m, inp))

def collect(loader):
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            _ = model(x)
        f = buf['x']
        feats.append(f)
        labels.append(y)
    return torch.cat(feats, 0).numpy(), torch.cat(labels, 0).numpy()

loader_orig = get_loader('NTU')
loader_rtg = get_loader('NTU_ID')

X_orig, Y_orig = collect(loader_orig)
X_rtg, Y_rtg = collect(loader_rtg)

X = np.concatenate([X_orig, X_rtg], 0)
D = np.concatenate([np.zeros(len(X_orig)), np.ones(len(X_rtg))], 0)

tsne = TSNE(n_components=2, perplexity=30, random_state=0)
Z = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
u = np.unique(Y_orig)
cmap = plt.cm.get_cmap('tab20', len(u))
for i, a in enumerate(u):
    idx = (np.arange(len(X_orig)))[Y_orig == a]
    plt.scatter(Z[idx, 0], Z[idx, 1], s=6, alpha=0.7, color=cmap(i))
plt.scatter(Z[len(X_orig):, 0], Z[len(X_orig):, 1], s=6, alpha=0.7, marker='x', color='k')
plt.tight_layout()
plt.savefig('tsne_action.png', dpi=200)

plt.figure(figsize=(8, 6))
idx0 = D == 0
idx1 = D == 1
plt.scatter(Z[idx0, 0], Z[idx0, 1], s=6, alpha=0.7, label='orig', color='#1f77b4')
plt.scatter(Z[idx1, 0], Z[idx1, 1], s=6, alpha=0.7, label='retarget', color='#ff7f0e', marker='x')
plt.legend()
plt.tight_layout()
plt.savefig('tsne_domain.png', dpi=200)

if len(np.unique(Y_orig)) > 1:
    sil = silhouette_score(X_orig, Y_orig, metric='euclidean')
    print('Silhouette by action (orig):', round(float(sil), 4))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_orig, Y_orig)
Y_rtg_pred = knn.predict(X_rtg)
match_rate = (Y_rtg_pred == Y_rtg).mean() if Y_rtg_pred.shape == Y_rtg.shape else np.nan
print('kNN transfer action match rate (if Y_rtg==action):', match_rate)

np.savez('domain_offset_features.npz',
         Z=Z, X=X, D=D, Y_orig=Y_orig, Y_rtg=Y_rtg, Y_rtg_pred=Y_rtg_pred)