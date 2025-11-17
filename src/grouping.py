# src/grouping.py
import numpy as np

def assign_households_to_transformers(
    n_households: int,
    households_per_tx: int,
) -> np.ndarray:
    """
    Returns an array of shape (n_households,) with transformer IDs.
    Transformer j gets households in a simple block assignment.
    """
    n_tx = int(np.ceil(n_households / households_per_tx))
    tx_ids = np.repeat(np.arange(n_tx), households_per_tx)[:n_households]
    return tx_ids
