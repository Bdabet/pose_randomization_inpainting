"""Minimal diffusion_policy-compatible Zarr ReplayBuffer writer."""
import numpy as np
import zarr
from numcodecs import Blosc


class ZarrReplayBufferWriter:
    """
    Minimal writer for the diffusion_policy ReplayBuffer zarr layout: a
    'data' group of equal-length (N, ...) arrays plus a 'meta/episode_ends'
    array of per-episode cumulative frame counts. Episodes are appended one
    at a time; array/chunk creation is lazy, inferred from the first episode.
    """
    COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    def __init__(self, path: str):
        self.root = zarr.open(path, mode="a")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")
        if "episode_ends" not in self.meta:
            self.meta.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype="i8")

    def add_episode(self, episode_data: dict) -> None:
        lengths = {len(v) for v in episode_data.values()}
        assert len(lengths) == 1, f"all arrays in an episode must share length N, got {lengths}"
        n_new = lengths.pop()

        for key, value in episode_data.items():
            value = np.asarray(value)
            if key not in self.data:
                chunk_len = min(100, max(1, n_new))
                self.data.create_dataset(
                    key, shape=(0,) + value.shape[1:], chunks=(chunk_len,) + value.shape[1:],
                    dtype=value.dtype, compressor=self.COMPRESSOR,
                )
            self.data[key].append(value, axis=0)

        prev_end = self.meta["episode_ends"][-1] if len(self.meta["episode_ends"]) else 0
        self.meta["episode_ends"].append(np.array([prev_end + n_new], dtype="i8"))
