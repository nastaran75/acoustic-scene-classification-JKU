
from tut17_nin_full import *
from asc.utils.data_tut17 import prepare, prepare_cyclic_shift


def get_valid_batch_iterator():
    """
    Get batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=None, shuffle=shuffle)

    return batch_iterator


def get_train_batch_iterator():
    """
    Get batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare_cyclic_shift, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


def get_train_strategy():
    return TrainingStrategy(
        batch_size=BATCH_SIZE,
        ini_learning_rate=INI_LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        L2=L2,
        adapt_learn_rate=get_constant(),
        update_function=get_update_adam(),
        valid_batch_iter=get_valid_batch_iterator(),
        train_batch_iter=get_train_batch_iterator(),
        best_model_by_accurary=True)

train_strategy = get_train_strategy()
