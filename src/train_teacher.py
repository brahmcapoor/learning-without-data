import os
import tensorflow as tf
import numpy as np
import sys
from models.teacher_model import TeacherModel


DATA_PATH = "../data/synthetic_data/first_dataset"


def train_and_save_model(model, folder, num_epochs):
    inputs = np.load(os.path.join(
        DATA_PATH, "inputs.npz")
    ).reshape(-1, 1)
    targets = np.load(os.path.join(
        DATA_PATH, "targets.npz")
    ).reshape(-1, 1)
    model.train(
        inputs=inputs,
        targets=targets,
        epochs=num_epochs
    )
    model.dump(os.path.join("../saved_models", folder, "model"))


def load_model(model, folder):
    model.load(os.path.join("../saved_models", folder))
    print(model.num_weights)
    print(model.get_weights)


if __name__ == "__main__":

    # TODO argparse
    assert len(sys.argv) >= 3, "Use python3 --load/--train <model_dir>"

    model = TeacherModel(
        input_dim=1,
        target_dim=1,
        layers=[12, 12, 8],
        activation=tf.nn.sigmoid,
        lr=1e-4
    )

    if sys.argv[1] == "--train":
        train_and_save_model(model, sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "--load":
        load_model(model, sys.argv[2])
    else:
        print("Invalid command")
