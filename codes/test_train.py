# Train MLP and/or CNN on MNIST with matched settings for Part B comparison.
import argparse
import gzip
import pickle
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
    with open("idx.pickle", "wb") as f:
        pickle.dump(idx, f)

    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    train_imgs = train_imgs.astype(np.float64) / 255.0
    valid_imgs = valid_imgs.astype(np.float64) / 255.0
    # 用训练集统计量做标准化，避免信息泄漏
    mean = train_imgs.mean()
    std = train_imgs.std() + 1e-8
    train_imgs = (train_imgs - mean) / std
    valid_imgs = (valid_imgs - mean) / std
    return train_imgs, train_labs, valid_imgs, valid_labs, rows, cols


def images_to_nchw(flat, rows, cols):
    """[N, rows*cols] -> [N, 1, rows, cols] for CNN."""
    n = flat.shape[0]
    return flat.reshape(n, 1, rows, cols)


def train_one_model(
    model,
    train_set,
    dev_set,
    save_dir,
    num_epochs=3,
    log_iters=100,
    max_iterations_per_epoch=1000,
):
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5
    )
    n_class = int(train_set[1].max()) + 1
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=n_class)
    runner = nn.runner.RunnerM(
        model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32, scheduler=scheduler
    )
    runner.train(
        train_set,
        dev_set,
        num_epochs=num_epochs,
        log_iters=log_iters,
        save_dir=save_dir,
        max_iterations_per_epoch=max_iterations_per_epoch,
    )
    return runner


def main():
    parser = argparse.ArgumentParser(description="MNIST MLP / CNN training (Part A & B)")
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn", "both"],
        default="both",
        help="Which model to train (default: both for fair comparison).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-iters-per-epoch",
        type=int,
        default=1000,
        help="Cap batches per epoch (default 1000; full pass is ~1563 for 50k/bs32).",
    )
    parser.add_argument("--log-iters", type=int, default=100)
    args = parser.parse_args()

    train_images_path = r".\dataset\MNIST\train-images-idx3-ubyte.gz"
    train_labels_path = r".\dataset\MNIST\train-labels-idx1-ubyte.gz"

    train_flat, train_labs, valid_flat, valid_labs, rows, cols = load_mnist_train_val(
        train_images_path, train_labels_path
    )

    # Same optimization setup: lr, batch_size=32, MultiStepLR, weight_decay 1e-4 on all trainable layers.
    wd = 1e-4
    runners = {}

    if args.model in ("mlp", "both"):
        print("=== Training MLP baseline ===")
        mlp = nn.models.Model_MLP(
            [train_flat.shape[-1], 600, 10], "ReLU", [wd, wd]
        )
        runners["mlp"] = train_one_model(
            mlp,
            [train_flat, train_labs],
            [valid_flat, valid_labs],
            save_dir=r"./best_models_mlp",
            num_epochs=args.epochs,
            log_iters=args.log_iters,
            max_iterations_per_epoch=args.max_iters_per_epoch,
        )
        print(f"[MLP] best validation accuracy: {runners['mlp'].best_score:.5f}")

    if args.model in ("cnn", "both"):
        print("=== Training CNN (same epochs / batch / lr / scheduler as MLP) ===")
        cnn = nn.models.Model_CNN(nn.models.build_mnist_cnn(weight_decay_lambda=wd))
        train_nchw = images_to_nchw(train_flat, rows, cols)
        valid_nchw = images_to_nchw(valid_flat, rows, cols)
        runners["cnn"] = train_one_model(
            cnn,
            [train_nchw, train_labs],
            [valid_nchw, valid_labs],
            save_dir=r"./best_models_cnn",
            num_epochs=args.epochs,
            log_iters=args.log_iters,
            max_iterations_per_epoch=args.max_iters_per_epoch,
        )
        print(f"[CNN] best validation accuracy: {runners['cnn'].best_score:.5f}")

    if args.model == "both":
        print("\n--- Part B comparison (matched training recipe) ---")
        print(f"MLP best val acc: {runners['mlp'].best_score:.5f}")
        print(f"CNN best val acc: {runners['cnn'].best_score:.5f}")

    # Learning curves
    n_modes = len(runners)
    if n_modes == 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        name = next(iter(runners.keys()))
        plot(runners[name], axes)
        fig.suptitle(f"{name.upper()} — loss & accuracy")
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plot(runners["mlp"], axes[0])
        plot(runners["cnn"], axes[1])
        axes[0, 0].set_title("MLP: loss")
        axes[0, 1].set_title("MLP: accuracy")
        axes[1, 0].set_title("CNN: loss")
        axes[1, 1].set_title("CNN: accuracy")
        fig.suptitle("MLP vs CNN (same lr, batch size, epochs, scheduler, data split)")
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
