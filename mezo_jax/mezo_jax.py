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
