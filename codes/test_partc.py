"""
Part C: two additional directions (course 1.4).

Direction 1 — Optimization: vanilla SGD vs SGD + momentum (MomentGD), same MLP and schedule.
Direction 2 — Regularization: MLP without dropout vs MLP with hidden dropout, same SGD and schedule.

Run from the codes/ directory:
  python test_partc.py --epochs 5
  python test_partc.py --direction momentum --epochs 3
  python test_partc.py --direction dropout --epochs 3
"""
import argparse
import gzip
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot


def load_mnist_train_val(train_images_path, train_labels_path, seed=309):
    np.random.seed(seed)
    with gzip.open(train_images_path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(train_labels_path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)
    idx = np.random.permutation(np.arange(num))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]
    scale = float(train_imgs.max())
    train_imgs = train_imgs.astype(np.float64) / scale
    valid_imgs = valid_imgs.astype(np.float64) / scale
    return train_imgs, train_labs, valid_imgs, valid_labs


def train_mlp(
    model,
    train_set,
    dev_set,
    optimizer,
    save_dir,
    num_epochs=5,
    log_iters=100,
):
    scheduler = nn.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5
    )
    n_class = int(train_set[1].max()) + 1
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=n_class)
    runner = nn.runner.RunnerM(
        model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32, scheduler=scheduler
    )
    runner.train(
        train_set, dev_set, num_epochs=num_epochs, log_iters=log_iters, save_dir=save_dir
    )
    return runner


def main():
    parser = argparse.ArgumentParser(description="Part C: momentum vs SGD, dropout vs no dropout")
    parser.add_argument(
        "--direction",
        choices=["both", "momentum", "dropout"],
        default="both",
        help="Which ablation to run (default: both).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log-iters", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9, help="MomentGD mu.")
    parser.add_argument("--dropout-p", type=float, default=0.5, help="Hidden dropout probability.")
    args = parser.parse_args()

    train_images_path = r".\dataset\MNIST\train-images-idx3-ubyte.gz"
    train_labels_path = r".\dataset\MNIST\train-labels-idx1-ubyte.gz"
    train_X, train_y, dev_X, dev_y = load_mnist_train_val(train_images_path, train_labels_path)

    in_dim = train_X.shape[1]
    wd = 1e-4
    runners = {}
    dirs = {}

    if args.direction in ("both", "momentum"):
        print("=== Part C.1 Optimization: SGD (baseline) vs MomentGD ===")
        m_sgd = nn.models.Model_MLP([in_dim, 600, 10], "ReLU", [wd, wd], hidden_dropout=0.0)
        opt_sgd = nn.optimizer.SGD(init_lr=0.06, model=m_sgd)
        runners["sgd"] = train_mlp(
            m_sgd,
            [train_X, train_y],
            [dev_X, dev_y],
            opt_sgd,
            save_dir=r"./partc_mlp_sgd",
            num_epochs=args.epochs,
            log_iters=args.log_iters,
        )
        print(f"[Baseline SGD] best val acc: {runners['sgd'].best_score:.5f}")

        m_mom = nn.models.Model_MLP([in_dim, 600, 10], "ReLU", [wd, wd], hidden_dropout=0.0)
        opt_mom = nn.optimizer.MomentGD(init_lr=0.06, model=m_mom, mu=args.momentum)
        runners["momentum"] = train_mlp(
            m_mom,
            [train_X, train_y],
            [dev_X, dev_y],
            opt_mom,
            save_dir=r"./partc_mlp_momentum",
            num_epochs=args.epochs,
            log_iters=args.log_iters,
        )
        print(f"[MomentGD mu={args.momentum}] best val acc: {runners['momentum'].best_score:.5f}")
        dirs["momentum"] = (
            "Same MLP [784,600,10], ReLU, weight_decay, lr, batch, MultiStepLR; "
            "only optimizer changes (SGD vs momentum)."
        )

    if args.direction in ("both", "dropout"):
        print("\n=== Part C.2 Regularization: no dropout vs dropout after hidden ReLU ===")
        if args.direction == "both" and "sgd" in runners:
            # Same recipe as Part C.1 SGD baseline — reuse curves (no second redundant run).
            runners["no_dropout"] = runners["sgd"]
            print(
                f"[No dropout baseline = Part C.1 SGD] best val acc: {runners['no_dropout'].best_score:.5f}"
            )
        else:
            m_no = nn.models.Model_MLP([in_dim, 600, 10], "ReLU", [wd, wd], hidden_dropout=0.0)
            opt_no = nn.optimizer.SGD(init_lr=0.06, model=m_no)
            runners["no_dropout"] = train_mlp(
                m_no,
                [train_X, train_y],
                [dev_X, dev_y],
                opt_no,
                save_dir=r"./partc_mlp_no_dropout",
                num_epochs=args.epochs,
                log_iters=args.log_iters,
            )
            print(f"[No dropout] best val acc: {runners['no_dropout'].best_score:.5f}")

        m_do = nn.models.Model_MLP(
            [in_dim, 600, 10], "ReLU", [wd, wd], hidden_dropout=args.dropout_p
        )
        opt_do = nn.optimizer.SGD(init_lr=0.06, model=m_do)
        runners["dropout"] = train_mlp(
            m_do,
            [train_X, train_y],
            [dev_X, dev_y],
            opt_do,
            save_dir=r"./partc_mlp_dropout",
            num_epochs=args.epochs,
            log_iters=args.log_iters,
        )
        print(f"[Dropout p={args.dropout_p}] best val acc: {runners['dropout'].best_score:.5f}")
        dirs["dropout"] = (
            "Same MLP width and SGD schedule; dropout (inverted) only after the first hidden ReLU; "
            "Runner disables dropout during validation."
        )

    # Figures
    n = len(runners)
    if n == 2 and args.direction == "momentum":
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plot(runners["sgd"], axes[0])
        plot(runners["momentum"], axes[1])
        fig.suptitle("Part C.1: SGD vs Momentum (matched setting)")
        plt.tight_layout()
    elif n == 2 and args.direction == "dropout":
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plot(runners["no_dropout"], axes[0])
        plot(runners["dropout"], axes[1])
        fig.suptitle("Part C.2: No dropout vs dropout (matched setting)")
        plt.tight_layout()
    elif args.direction == "both" and "dropout" in runners:
        fig, axes = plt.subplots(4, 2, figsize=(10, 14))
        plot(runners["sgd"], axes[0])
        plot(runners["momentum"], axes[1])
        plot(runners["no_dropout"], axes[2])
        plot(runners["dropout"], axes[3])
        axes[0, 0].set_title("C.1 SGD loss")
        axes[0, 1].set_title("C.1 SGD acc")
        axes[1, 0].set_title("C.1 Momentum loss")
        axes[1, 1].set_title("C.1 Momentum acc")
        axes[2, 0].set_title("C.2 No dropout (same run as C.1 SGD)")
        axes[2, 1].set_title("C.2 No dropout acc")
        axes[3, 0].set_title("C.2 Dropout loss")
        axes[3, 1].set_title("C.2 Dropout acc")
        fig.suptitle("Part C: optimization + regularization ablations")
        plt.tight_layout()

    if dirs:
        print("\n--- Report notes (controlled factors) ---")
        for k, v in dirs.items():
            print(f"[{k}] {v}")

    plt.show()


if __name__ == "__main__":
    main()
