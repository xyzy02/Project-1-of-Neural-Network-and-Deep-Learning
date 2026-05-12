### Start Up

First look into the `dataset_explore.ipynb` and get familiar with the data.

### Codes need your implementation

1. `op.py` 
   Implement the forward and backward function of `class Linear`
   Implement the `MultiCrossEntropyLoss`. Note that the `Softmax` layer could be included in the `MultiCrossEntropyLoss`.
   Try to implement `conv2D`, do not worry about the efficiency.
   You're welcome to implement other complicated layer (e.g.  ResNet Block or Bottleneck)
2. `models.py` You may freely edit or write your own model structure.
3. `mynn/lr_scheduler.py` You may implement different learning rate scheduler in it.
4. `MomentGD` in `optimizer.py`
5. Modifications in `runner.py` if needed when your model structure is slightly different from the given example.


### Train the model.

Open test_train.py, modify parameters and run it.

If you want to train the model on your own dataset, just change the values of variable *train_images_path* and *train_labels_path*

### Test the model.

Open test_model.py, specify the saved model's path and the test dataset's path, then run the script, the script will output the accuracy on the test dataset.

### Course submission: GitHub link and checkpoints (do not upload large files to Git)

1. **Create a public GitHub repository** and push only source code (e.g. `mynn/`, scripts, `report.md`). The repo root `PJ1/.gitignore` excludes `*.gz` under `dataset/`, `*.pickle`, and `best_models*` folders so datasets and weights are not tracked by default.
2. **In your submitted write-up / report**, add one line with the **GitHub repo URL** (e.g. `https://github.com/<user>/<repo>`).
3. **Weights must not live in that repo** if the course forbids large uploads. Instead:
   - Zip `best_models_mlp/best_model.pickle`, `best_models_cnn/best_model.pickle`, etc. locally.
   - Upload the zip to **Google Drive / OneDrive / 百度网盘 / Hugging Face Hub / Zenodo** (or similar), set sharing to “anyone with the link” (or as required), and put that **download link** in the report, with a one-line note on which archive corresponds to which experiment.
4. **README for TAs**: In the report (or repo `README`), state how to obtain MNIST (official site or mirror), where to place files under `codes/dataset/MNIST/`, and how to run `test_train.py` to reproduce checkpoints if needed.

