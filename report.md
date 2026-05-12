# PJ1

章一诺

23307110124

---

## 1. MLP baseline（Part A）

在 Part A 中，我们实现了一个基础的多层感知机（MLP）用于 MNIST 手写数字分类。核心算子包括手写的线性层 `Linear` 和多类交叉熵损失 `MultiCrossEntropyLoss`。线性层前向计算为 \(Y = XW + b\)。`MultiCrossEntropyLoss` 对 batch 使用均值交叉熵，并在前向中给出对 logits 的梯度 \(G = (P - Y_{\mathrm{onehot}})/N\)（\(N\) 为 batch 大小）。`Linear.backward` 中 \(\partial W = X^\top G\)，\(\partial b\) 为对 \(G\) 在 batch 维求和，\(\partial X = G W^\top\)，不再额外除以 \(N\)，与上述损失梯度衔接一致。多类交叉熵结合 Softmax 实现了数值稳定的前向计算，并在 `backward` 中调用模型的 `backward` 方法完成反向传播，使得网络参数能够按照梯度正确更新。

模型结构采用 `Model_MLP([784, 600, 10], 'ReLU')`，隐藏层使用 ReLU 激活函数，这种设置可以提供非线性表达能力，同时避免梯度消失问题。训练时使用 SGD 优化器，初始学习率设置为 0.06，batch size 为 32，这一选择在初期能够保持较快的收敛速度，同时较小 batch 有助于梯度估计的噪声提供一定正则化效果。权重衰减系数为 1e-4，用于抑制模型过拟合，同时与课程 starter 代码保持一致。数据划分方面，在 `test_train.py` 中对训练集索引做随机打乱后，取其中 10000 个样本作为验证集，其余约 50000 个样本用于训练。像素先除以 255 映射到 [0,1]，再使用训练子集的均值与标准差对训练集与验证集做相同线性标准化（验证集使用训练子集的统计量，避免信息泄漏）。另外观察训练损失曲线发现，其实模型在较早时间已经几乎收敛，因此在 `test_train.py` 中将训练轮数设为 3 个 epoch，并通过 `max_iterations_per_epoch=1000` 限制每个 epoch 内的参数更新次数，训练效果基本上没有损失。

训练过程中，模型在每个 iteration 记录训练和验证的 loss 与 accuracy，通过 `RunnerM` 和绘图工具生成学习曲线。图 1 展示了 MLP 在训练过程中的 loss 和 accuracy 曲线，曲线呈现稳定下降趋势，并在后期趋于平稳，验证集准确率逐渐收敛，反映出 MLP 模型在该配置下已达到较为合理的性能水平。

![图 1：MLP 与 CNN 学习曲线对比](./Figure_1.png)

---

## 2. CNN 模型与 MLP–CNN 对比（Part B）

在 Part B 中，我们实现了一个简单的卷积神经网络（CNN）以充分利用图像的空间信息。卷积算子 `conv2D` 在 `mynn/op.py` 中实现，前向计算通过在输入特征图上滑动卷积核求和并加上偏置完成；反向传播时根据链式法则在 batch 与空间位置上累加 \(\partial W\)、\(\partial b\) 及对输入的梯度。由于上游损失已对 batch 取平均，传入的 `grads` 已含 \(1/N\) 因子，本层不再对 batch 重复除以 \(N\)，从而在整条链上与 mean 交叉熵的梯度尺度一致。

CNN 模型由 `build_mnist_cnn` 构建：第一层卷积输入通道 1、输出通道 8，卷积核 5×5，stride 为 2，ReLU 激活；第二层卷积输入 8、输出通道 16，卷积核 5×5，stride 为 2，ReLU 激活。空间尺寸由 28×28 经两次 valid 卷积与步幅下采样变为 12×12 再至 4×4，展平后维度为 \(16 \times 4 \times 4 = 256\)。分类头使用 `Linear(256, 10)` 输出 10 个类别。Flatten 操作由 `mynn/op.py` 中的 `Flatten` 完成，将四维张量 `[N,C,H,W]` 转为 `[N,C·H·W]`。卷积层和全连接层均使用与 MLP 相同的权重衰减参数，以保持公平对比。

在对比实验中，我们尽量保持训练设置一致，以遵循「一次只改变一个主要因素」原则。优化器、初始学习率、batch size、学习率调度里程碑以及训练轮数均与 MLP 保持一致，输入形状调整为 `[N,1,28,28]`。图 1 下行显示 CNN 的 loss 与 accuracy 曲线，相比 MLP，其收敛更快，验证准确率更高，充分体现了卷积结构在图像分类中对局部空间特征的优势。

---

## 3. 两个扩展方向（Part C）

在 Part C 中，我们选择了优化和正则化两个方向进行扩展实验。优化方向采用带动量的 MomentGD，反向传播逻辑与 SGD 相同，但在每次参数更新时引入动量项 \(v \leftarrow \mu v + g\)，并使用 \(\theta \leftarrow \theta - \eta v\) 更新参数（与 `mynn/optimizer.py` 中实现一致），这样可以加速收敛并减少训练中震荡。`test_partc.py` 中 MLP 仍为 `[784, 600, 10]`、ReLU、SGD 初始学习率 0.06、batch size 32、`MultiStepLR` 与 Part A/B 相同。数据预处理在 `test_partc.py` 中为像素除以 255 得到 [0,1]，未使用 `test_train.py` 中的训练集 z-score，若需与 Part A/B 严格可比，应在脚本层统一预处理。Dropout 在 `Model_MLP` 中置于每个中间隐藏块的 ReLU 之后（Inverted Dropout；当前网络仅一层隐藏 ReLU，故等价于在该层后插入）；`RunnerM` 在验证时将模型设为 eval，Dropout 关闭。

---

## 4. Main results

各实验结果如下：图 2 为 Part C 多组实验总览；图 3 为 SGD 与 MomentGD 的 loss 与 accuracy 对比；图 4 为无 Dropout 与有 Dropout 的对比。动量实验相比标准 SGD 往往收敛更快、验证准确率略有提升；Dropout 实验中训练 loss 可能略升而验证表现改善，体现对过拟合的抑制。

![图 2：Part C 扩展实验总览](./Figure_2.png)  
![图 3：SGD 与 MomentGD 学习曲线](./Figure_3.png)  
![图 4：无 Dropout 与 Dropout 学习曲线](./Figure_4.png)

---

## 5. Detailed visualization

loss 曲线可用于观察训练和验证的收敛情况以及过拟合趋势，accuracy 曲线帮助对比 train 和 dev 之间的差距，从而评价泛化能力。CNN 与 MLP、动量与无动量、Dropout 开启与关闭的对比均可通过学习曲线直观体现。具体的，可以通过 `weight_visualization.py` 进行权重的可视化，结果如下：

![图 5：权重可视化](./Figure_2.png)  

---

## 6. 讨论

Q1: CNN 在图像分类中通常优于 MLP，原因在于卷积层能够利用局部感受野和权值共享来编码图像的空间结构，从而更高效地提取平移不变特征。相比之下，MLP 将图像拉直后丢失了像素间的空间关系，需要更多参数才能近似同样的映射。

Q2: 实验结果显示，CNN 相比 MLP 在验证集上取得了明显提升。学习曲线表明 CNN 在训练中收敛速度更快，最终验证准确率通常高于 MLP，且曲线波动可相对更小，说明在该设置下泛化更好。主训练脚本 `test_train.py` 以验证集监控为主；若使用课程提供的测试集评估脚本对 MNIST 测试集再测，可进一步核对与验证集结论是否一致。整体来看，CNN 的结构改进有利于在未见样本上取得更好表现。

Q3: 在 Part C 中，选择优化和正则化两个方向。优化方向采用带动量的 MomentGD，以提升收敛速度并平滑梯度轨迹，使训练更加稳定；通过对比 SGD 和 MomentGD 的学习曲线，可以清晰观察动量对收敛动态的影响。而为了应对过拟合的情形，另选正则化方向采用 Dropout 方法：在 `Model_MLP` 实现中位于各中间隐藏 ReLU 之后（Inverted Dropout），训练阶段生效，验证阶段由 `RunnerM` 关闭 Dropout，从而增强模型泛化能力。另外这两个方向实现成本低、易于与基线对比，并且直接对应了深度学习中核心的优化与过拟合控制问题，便于进行清晰的实验分析。

Q4: MLP 与 CNN 的对比最有效，因为它直接展示了模型结构对图像特征提取和泛化能力的影响。可以观察到 CNN 的收敛速度更快且验证准确率更高，体现了卷积归纳偏置的实际效果。在 Part C 中，动量优化的主要信息量体现在收敛轨迹的平滑和速度提升；Dropout 正则化的主要信息量体现在训练与验证差距缩小、过拟合得到抑制。

Q5: 对于几何形状相似的图片数据容易出现分类错误的现象，尤其是存在共享局部部分或环结构的情形。


## 7. 代码地址

GitHub仓库：https://github.com/xyzy02/Project-1-of-Neural-Network-and-Deep-Learning

模型权重：https://pan.baidu.com/s/1Nis5X8EPMnIJsAKAyHThWg 提取码: rcy5
