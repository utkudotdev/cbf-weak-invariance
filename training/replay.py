"""Device-resident general and unsafe-state replay for labeled graph states."""

from functools import partial

import jax
import jax.numpy as jnp


def concatenate(datasets):
    """Join states, safe labels, and unsafe labels along their sample axis."""
    return tuple(jnp.concatenate(leaves, axis=0) for leaves in zip(*datasets))


@partial(jax.jit, static_argnames=("count",))
def sample(data, size, count: int, key: jax.Array):
    """Sample with replacement from a nonempty buffer."""
    indices = jax.random.randint(key, (count,), 0, size)
    return tuple(x[indices] for x in data)


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")

        self.capacity = capacity
        self.data = None

    def __len__(self):
        return 0 if self.data is None else self.data[0].shape[0]

    def append(self, data, mask=None):
        data = tuple(jnp.asarray(x) for x in data)
        if mask is not None:
            # Finding the selection length may synchronize a scalar with the
            # host; the indices, states, and labels remain on device.
            indices = jnp.flatnonzero(mask)
            data = tuple(x[indices] for x in data)

        if data[0].shape[0] == 0:
            return

        if self.data is not None:
            data = concatenate((self.data, data))

        self.data = tuple(x[-self.capacity :] for x in data)

    def sample(self, count: int, key: jax.Array):
        if not len(self):
            raise ValueError("cannot sample an empty replay buffer")

        return sample(self.data, len(self), count, key)


class ReplaySampler:
    def __init__(self, capacity: int):
        self.general = ReplayBuffer(capacity)
        self.unsafe = ReplayBuffer(max(1, capacity // 2))

    def mix(self, fresh, key: jax.Array):
        """Mix equal fresh/general/unsafe counts after the first collection.

        Replay is drawn with replacement from previous collections. If no
        unsafe state has been seen, use general replay for that third instead.
        Labels remain attached to the trajectories that originally produced them.
        """
        fresh = tuple(jnp.asarray(x) for x in fresh)
        count = fresh[0].shape[0]
        datasets = [fresh]

        if len(self.general):
            general_key, unsafe_key = jax.random.split(key)
            unsafe_source = self.unsafe if len(self.unsafe) else self.general
            datasets.extend(
                (
                    self.general.sample(count, general_key),
                    unsafe_source.sample(count, unsafe_key),
                )
            )

        # Store only fresh training data, after sampling the historical buffers.
        self.general.append(fresh)
        self.unsafe.append(fresh, jnp.any(fresh[2], axis=-1))

        return concatenate(datasets)
