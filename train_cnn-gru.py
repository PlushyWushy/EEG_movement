
"""
CNN (~85% test accuracy)
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from train_mlp import (
    CLASSES,
    N_SAMPLES,
    build_dataset,
    run_epoch,
    smote_augment,
    subject_dependent_split,
    subject_independent_split,
)


class CNN1D(nn.Module):

    def __init__(self, n_channels, n_times, num_classes,
                 n_filters1=32, n_filters2=32,
                 kernel_size1=20, kernel_size2=6,
                 gru_hidden=128,
                 fc_dim=64, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, n_filters1, kernel_size1, padding="same"),
            nn.BatchNorm1d(n_filters1),
            nn.ReLU(),

            nn.Conv1d(n_filters1, n_filters2, kernel_size1, padding="valid"),
            nn.BatchNorm1d(n_filters2),
            nn.ReLU(),
            nn.Dropout1d(dropout),

            nn.Conv1d(n_filters2, n_filters2, kernel_size2, padding="valid"),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(n_filters2, n_filters2, kernel_size2, padding="valid"),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )
        self.gru = nn.GRU(input_size=n_filters2, hidden_size=gru_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, n_channels, n_times)
        x = self.features(x)              # (batch, n_filters2, time')
        x = x.transpose(1, 2)             # (batch, time', n_filters2) -- GRU wants time-major
        _, h_n = self.gru(x)              # h_n: (1, batch, gru_hidden), final hidden state
        x = h_n.squeeze(0)                # (batch, gru_hidden)
        return self.classifier(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-subjects", type=int, default=None,
                     help="Limit number of subjects processed (default: all available)")
    ap.add_argument("--no-cache", action="store_true",
                     help="Ignore/overwrite the cached processed-epoch array")
    ap.add_argument("--n-filters1", type=int, default=32)
    ap.add_argument("--n-filters2", type=int, default=32)
    ap.add_argument("--kernel-size1", type=int, default=20,
                     help="Kernel size for conv1/conv2 (CNN-GRU paper: 20)")
    ap.add_argument("--kernel-size2", type=int, default=6,
                     help="Kernel size for conv3/conv4 (CNN-GRU paper: 6)")
    ap.add_argument("--gru-hidden", type=int, default=128,
                     help="GRU hidden size (CNN-GRU paper: 128)")
    ap.add_argument("--fc-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15,
                     help="Early-stopping patience (epochs without val-loss improvement)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["subject-dependent", "subject-independent"],
                     default="subject-dependent",
                     help="'subject-dependent' (default): every subject contributes to "
                          "train/val/test, matching the reviewed papers' protocol. "
                          "'subject-independent': whole subjects held out for val/test, "
                          "testing generalization to unseen users.")
    ap.add_argument("--no-smote", action="store_true",
                     help="Disable SMOTE oversampling of minority classes in the training set")
    ap.add_argument("--smote-k", type=int, default=5,
                     help="Nearest neighbors used by SMOTE (CNN-GRU paper default: 5)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, groups = build_dataset(max_subjects=args.max_subjects,
                                  use_cache=not args.no_cache)
    print(f"Dataset: X={X.shape}, classes={CLASSES}, "
          f"subjects={len(set(groups.tolist()))}")
    print("Class counts:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})

    if args.split_mode == "subject-dependent":
        train_mask, val_mask, test_mask = subject_dependent_split(
            y, groups, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
    else:
        train_mask, val_mask, test_mask = subject_independent_split(
            groups, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Epochs: train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    # Normalization is done per-subject at load time (see load_subject_epochs),
    # matching the paper's "normalizing within participants" -- no global
    # train-set normalization step needed here.

    if not args.no_smote:
        X_train, y_train = smote_augment(X_train, y_train, k=args.smote_k, seed=args.seed)
        counts = {c: int((y_train == i).sum()) for i, c in enumerate(CLASSES)}
        print(f"After SMOTE: train={len(y_train)} epochs, class counts={counts}")

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    print(f"Using device: {device}")

    def make_loader(Xa, ya, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xa).float(), torch.from_numpy(ya).long()
        )
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    assert n_times == N_SAMPLES
    model = CNN1D(n_channels, n_times, len(CLASSES),
                  n_filters1=args.n_filters1, n_filters2=args.n_filters2,
                  kernel_size1=args.kernel_size1, kernel_size2=args.kernel_size2,
                  gru_hidden=args.gru_hidden,
                  fc_dim=args.fc_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nTest loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    print("\nClassification report (test set):")
    print(classification_report(all_true, all_preds, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(CLASSES)
    print(confusion_matrix(all_true, all_preds))


if __name__ == "__main__":
    main()
