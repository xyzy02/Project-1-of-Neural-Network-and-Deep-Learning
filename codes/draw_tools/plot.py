# plot the score and loss
import numpy as np
import matplotlib.pyplot as plt

colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def _dev_iteration_x(n_train_steps, n_dev_points):
    """Map each dev metric (one per epoch) to the iteration index at epoch end."""
    if n_dev_points == 0:
        return np.array([], dtype=int)
    if n_dev_points == n_train_steps:
        return np.arange(n_dev_points, dtype=int)
    # e.g. 3000 train steps, 3 dev points -> x = 999, 1999, 2999
    return np.clip(
        np.round((np.arange(1, n_dev_points + 1) * n_train_steps / n_dev_points) - 1).astype(int),
        0,
        max(n_train_steps - 1, 0),
    )

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]
    
    n_train = len(runner.train_scores)
    iters = np.arange(n_train)
    # 绘制训练损失变化曲线
    axes[0].plot(iters, runner.train_loss, color=train_color, label="Train loss")
    # 绘制评价损失变化曲线（可能按 epoch 记录，长度与 train 不一致）
    n_dev = len(runner.dev_loss)
    if n_dev > 0:
        axes[0].plot(
            _dev_iteration_x(n_train, n_dev),
            runner.dev_loss,
            color=dev_color,
            linestyle="--",
            label="Dev loss",
        )
    # 绘制坐标轴和图例
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("")
    axes[0].legend(loc='upper right')
    # 绘制训练准确率变化曲线
    axes[1].plot(iters, runner.train_scores, color=train_color, label="Train accuracy")
    # 绘制评价准确率变化曲线
    if len(runner.dev_scores) > 0:
        axes[1].plot(
            _dev_iteration_x(n_train, len(runner.dev_scores)),
            runner.dev_scores,
            color=dev_color,
            linestyle="--",
            label="Dev accuracy",
        )
    # 绘制坐标轴和图例
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')