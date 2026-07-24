"""Formato de cable de los embeddings, compartido por definición con los otros dos servicios.

``base64(np.save(matriz float32))`` — el search-service lo decodifica con ``np.load`` y
espera un array 2D float32 con una fila por elemento entrenado, en orden de id ASC.
"""

from __future__ import annotations

import base64
import io

import numpy as np


def encode_matrix(matrix: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(matrix, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_matrix(encoded: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)
