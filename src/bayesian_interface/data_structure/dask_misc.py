def make_chunks(shape: tuple, on_chain: bool, on_dim: bool) -> tuple:
    chunks = []
    if on_chain:
        chunks.append(1)

    if on_chain and on_dim:
        step_ext = shape[1:-1]
    elif on_chain:
        step_ext = shape[1 : shape[-1] + 1]
    elif on_dim:
        step_ext = shape[0:-1]
    else:
        step_ext = shape
    chunks += list(step_ext)

    if on_dim:
        chunks.append(1)

    return tuple(chunks)
