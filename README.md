# JAX MeZO: Fine-Tuning Language Models with Just Forward Passes

JAX unofficial implementation of "Fine-Tuning Language Models with Just Forward
Passes"

> Disclaimer: this is a work-in-progress and I am still relatively
> inexperienced with Jax. There could be bugs or inefficiencies in the code,
> feel free to open an issue or pull request!

This library adds functional transforms that take an input function `fn(params,
...)` and transforms it into one that computes the approximate gradient with
respect to `params` via a memory-efficient, zeroth-order (MeZO) method. This
allows for fine-tuning of models using only model forward passes, significantly
reducing the memory required.

See [the original paper](https://arxiv.org/abs/2305.17333) for more details on
this method.

## Installation

Install using `pip install git+https://github.com/vvvm23/lora-jax` or navigate
to root of repository and run `pip install .`

## Usage

Suppose we have a function we want to compute an approximate gradient for:
```python
def fn(params, x):
    y = x @ params['w'] = params['b']
    return (y**2).mean()
```

We can create a new function that computes the MeZO gradient with respect to
the first argument by wrapping it in `mezo_jax.mezo_grad`:
```python
from mezo_jax import mezo_grad
grad_fn = mezo_grad(fn)
```

`grad_fn` takes as input the original arguments and two additional values: the
noise scale and a PRNG key. The noise scale controls how much to permute the
parameters by, and the PRNG key is used to generate random arrays.

```python
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0xffff)

params = {'w': jnp.ones((2,2)), 'b': jnp.zeros((2,))}

key, x_key = jax.random.split(key)
x = jax.random.normal(x_key, (2,))

scale = 1e-3
key, mezo_key = jax.random.split(key)
grad = grad_fn(params, scale, mezo_key, x)
```

This returns the projected gradient. To rematerialize the full gradient it must
be multiplied by the original noise array used to perturb the parameters.

We provide a helper function to apply the updates to the parameters:
```python
from mezo_jax import apply_updates
learning_rate = 1e-4
new_params = apply_updates(params, learning_rate*grad, mezo_key)
```

If you want to simultaneously return the gradient and the value of `fn` itself,
use `mezo_jax.mezo_value_and_grad`.

If you return auxiliary data from your function (such as accuracy) pass
`has_aux=True` to the `mezo_grad` or `mezo_value_and_grad` calls.

## Examples
See `examples/` for example usage in fine-tuning large language models.

---

### TODO
- [ ] specify arbitrary parameters to exclude from updates
- [ ] specify arbitrary parameters to compute gradient with respect to
- [ ] example showing decoder-only model fine-tuning
- [X] example showing encoder-only model fine-tuning
- [X] example showing training on non-differentiable metric (like accuracy)

