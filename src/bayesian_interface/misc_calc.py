import numpy as np


def split_nsteps(nsteps: int, nsteps_chunk: int) -> np.ndarray:
    """nstepsをチャンク刻みに分解する
    :param nsteps: MCMCチェーンの長さ
    :param nsteps_chunk: チャンクの長さ
    :return: 分割されたnstepsのリスト
    """
    if nsteps <= nsteps_chunk:
        chunks_nsteps = [nsteps]
    else:
        num_split = int(np.ceil(nsteps / nsteps_chunk))
        chunks_nsteps = []
        for i in range(num_split):
            if i == num_split - 1:
                value = nsteps - sum(chunks_nsteps)
            else:
                value = nsteps_chunk
            chunks_nsteps.append(value)

    return np.array(chunks_nsteps, dtype=np.int64)
