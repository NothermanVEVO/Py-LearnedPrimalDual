# import LearnedPrimalDual.LearnedPrimalDual as LearnedPrimalDual

# print("Creating model...")

# model = LearnedPrimalDual.learned_primal_dual_model(15)

# model.summary()

#################################################################################

QUANT_OF_TRAIN_IMGS = 2
X_TRAIN_PATH = "dataset/x_train"
Y_TRAIN_PATH = "dataset/y_train"

QUANT_OF_TEST_IMGS = 2
X_TEST_PATH = "dataset/x_test"
Y_TEST_PATH = "dataset/y_test"

PROJECTIONS = [90]

import os
import phantoms.Dataset as Dataset


def _generate_datasets() -> None:
    print("Generating TRAIN dataset...", QUANT_OF_TRAIN_IMGS)

    os.makedirs(X_TRAIN_PATH, exist_ok=True)
    os.makedirs(Y_TRAIN_PATH, exist_ok=True)

    Dataset.generate_custom_data_set(512,
        QUANT_OF_TRAIN_IMGS,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        PROJECTIONS
    )

    print("Generating TEST dataset...", QUANT_OF_TEST_IMGS)

    os.makedirs(X_TEST_PATH, exist_ok=True)
    os.makedirs(Y_TEST_PATH, exist_ok=True)

    Dataset.generate_custom_data_set(512,
        QUANT_OF_TEST_IMGS,
        X_TEST_PATH,
        Y_TEST_PATH,
        PROJECTIONS
    )

_generate_datasets()