import jax


def generate_key_tree(key: jax.random.PRNGKey, target=None, treedef=None):
    assert (target is None) != (treedef is None)
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)

    keys = jax.random.split(key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def random_normal_like_tree(key, target):
    key_tree = generate_key_tree(key, target=target)
    return jax.tree_map(lambda t, k: jax.random.normal(k, t.shape, dtype=t.dtype), target, key_tree)
