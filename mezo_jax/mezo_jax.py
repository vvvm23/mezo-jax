from functools import wraps

import jax

from .utils import generate_key_tree


def perturb_parameters(params, scale, key):
    key_tree = generate_key_tree(key, target=params)
    params = jax.tree_map(lambda p, k: p + scale * jax.random.normal(k, p.shape, dtype=p.dtype), params, key_tree)
    return params


def apply_updates(params, grad, key):
    return perturb_parameters(params, -grad, key)


def mezo_grad(loss_fn, has_aux: bool = False):
    @wraps(loss_fn)
    def _mezo_grad(params, scale, mezo_key, *args, **kwargs):
        params = perturb_parameters(params, scale, mezo_key)
        pos_out = loss_fn(params, *args, **kwargs)

        if has_aux:
            loss_pos, *aux_pos = pos_out
        else:
            loss_pos = pos_out

        params = perturb_parameters(params, -2 * scale, mezo_key)
        neg_out = loss_fn(params, *args, **kwargs)
        if has_aux:
            loss_neg, *aux_neg = neg_out
        else:
            loss_neg = neg_out

        params = perturb_parameters(params, scale, mezo_key)
        grad = (loss_pos - loss_neg) / (2 * scale)

        if has_aux:
            avg_aux = jax.tree_map(lambda p, n: (p + n) / 2, aux_pos, aux_neg)
            return grad, avg_aux

        return grad

    return _mezo_grad


def mezo_value_and_grad(loss_fn, has_aux: bool = False):
    @wraps(loss_fn)
    def _mezo_value_and_grad(params, scale, mezo_key, *args, **kwargs):
        params = perturb_parameters(params, scale, mezo_key)
        pos_out = loss_fn(params, *args, **kwargs)

        if has_aux:
            loss_pos, *aux_pos = pos_out
        else:
            loss_pos = pos_out

        params = perturb_parameters(params, -2 * scale, mezo_key)
        neg_out = loss_fn(params, *args, **kwargs)

        if has_aux:
            loss_neg, *aux_neg = neg_out
        else:
            loss_neg = neg_out

        params = perturb_parameters(params, scale, mezo_key)
        grad = (loss_pos - loss_neg) / (2 * scale)

        if has_aux:
            return ((loss_pos + loss_neg) / 2, *jax.tree_map(lambda p, n: (p + n) / 2, aux_pos, aux_neg)), grad

        return (loss_pos + loss_neg) / 2, grad

    return _mezo_value_and_grad


if __name__ == "__main__":
    import jax.numpy as jnp

    def loss_fn(params, x):
        h = x @ params["x"]
        h = h + params["y"]

        return ((x - h) ** 2).mean(), (h == x).mean()

    params = dict(x=jnp.ones((2, 2)), y=jnp.zeros(2))
    print(params)

    scale = 1e-2
    lr = 1e-3

    key = jax.random.PRNGKey(0)
    key, x_key = jax.random.split(key)
    data = jax.random.normal(x_key, (2,))

    grad_loss_fn = mezo_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(params, data, mezo_key):
        grad, (accuracy,) = grad_loss_fn(params, scale, mezo_key, data)
        scaled_grad = lr * grad
        new_params = apply_updates(params, scaled_grad, mezo_key)

        return grad, new_params, accuracy

    grad_avg = 0.0
    accuracy_avg = 0.0
    for i in range(10000):
        key, subkey = jax.random.split(key)
        grad, params, accuracy = train_step(params, data, subkey)

        grad_avg += grad
        accuracy_avg += accuracy
        if i % 100 == 0:
            print(grad_avg / 100, accuracy_avg / 100)
            grad_avg, accuracy_avg = 0.0, 0.0
