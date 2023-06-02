from functools import wraps

import jax

from .utils import generate_key_tree


# TODO: check if this is indeed done in place
def perturb_parameters(params, scale, key):
    key_tree = generate_key_tree(
        key, target=params
    )  # TODO: do we need to keep generating key tree? or can pass it around
    params = jax.tree_map(lambda p, k: p + scale * jax.random.normal(k, p.shape, dtype=p.dtype), params, key_tree)
    return params


def apply_updates(params, grad, key):
    return perturb_parameters(params, -grad, key)


def mezo_grad(loss_fn):
    @wraps(loss_fn)
    def _mezo_grad(params, scale, mezo_key, *args, **kwargs):
        params = perturb_parameters(params, scale, mezo_key)
        loss_pos = loss_fn(params, *args, **kwargs)

        params = perturb_parameters(params, -2 * scale, mezo_key)
        loss_neg = loss_fn(params, *args, **kwargs)

        params = perturb_parameters(params, scale, mezo_key)
        grad = (loss_pos - loss_neg) / (2 * scale)

        return grad

    return _mezo_grad


# TODO: only value is the loss itself, take the average
def mezo_value_and_grad(loss_fn):
    @wraps(loss_fn)
    def _mezo_value_and_grad(params, scale, mezo_key, *args, **kwargs):
        params = perturb_parameters(params, scale, mezo_key)
        loss_pos = loss_fn(params, *args, **kwargs)

        params = perturb_parameters(params, -2 * scale, mezo_key)
        loss_neg = loss_fn(params, *args, **kwargs)

        params = perturb_parameters(params, scale, mezo_key)
        grad = (loss_pos - loss_neg) / (2 * scale)

        return (loss_pos + loss_neg) / 2, grad

    return _mezo_value_and_grad


if __name__ == "__main__":
    import jax.numpy as jnp

    def loss_fn(params, x):
        h = x @ params["x"]
        h = h + params["y"]

        return ((x - h) ** 2).mean()

    params = dict(x=jnp.ones((2, 2)), y=jnp.zeros(2))
    print(params)

    scale = 1e-2
    lr = 1e-3

    key = jax.random.PRNGKey(0)
    key, x_key = jax.random.split(key)
    data = jax.random.normal(x_key, (2,))

    grad_loss_fn = mezo_grad(loss_fn)

    @jax.jit
    def train_step(params, data, mezo_key):
        grad = grad_loss_fn(params, scale, mezo_key, data)
        scaled_grad = lr * grad
        new_params = apply_updates(params, scaled_grad, mezo_key)

        return grad, new_params

    grad_avg = 0.0
    for i in range(10000):
        key, subkey = jax.random.split(key)
        grad, params = train_step(params, data, subkey)

        grad_avg += grad
        if i % 100 == 0:
            print(grad_avg / 100)
            grad_avg = 0.0
