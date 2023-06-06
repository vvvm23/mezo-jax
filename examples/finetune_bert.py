import argparse
from typing import Optional, Tuple, Union

import datasets
import jax
import jax.numpy as jnp
import optax
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

from mezo_jax import apply_updates, mezo_value_and_grad


def load_tokenizer(name: str = "roberta-large"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def load_dataset(
    tokenizer: AutoTokenizer,
    name: Union[str, Tuple[str, str]] = ("glue", "mnli"),
    batch_size: int = 16,
    num_workers: int = 4,
    max_length: Optional[int] = 128,
):
    if max_length is None:
        max_length = tokenizer.max_length
    if isinstance(name, str):
        name = (name,)
    dataset = datasets.load_dataset(*name)

    def tokenize_function(examples):
        label = examples.pop("label")
        return {
            "label": label,
            **tokenizer(*examples.values(), padding="max_length", truncation=True, max_length=max_length),
        }

    dataset = dataset.remove_columns("idx")
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    dataset.set_format("torch")

    train_dataloader = DataLoader(
        dataset["train"], shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True
    )
    # eval_dataloader = DataLoader(dataset["test"], shuffle=False, batch_size=batch_size, num_workers=num_workers)

    # return train_dataloader, eval_dataloader
    return train_dataloader, dataset["train"]


def load_pretrained_model(name: str = "roberta-large", num_labels: int = 3):
    return FlaxAutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)


def main(args):
    use_mezo = not args.disable_mezo
    dataset_name = args.dataset_name if args.subset is None else (args.dataset_name, args.subset)
    assert not args.optimise_accuracy or not args.disable_mezo, "Cannot optimise for accuracy without MeZo enabled"

    key = jax.random.PRNGKey(args.seed)
    tokenizer = load_tokenizer()
    train_dataloader, train_dataset = load_dataset(
        tokenizer, dataset_name, batch_size=args.batch_size, num_workers=args.num_workers, max_length=args.max_length
    )
    num_labels = train_dataset.features["label"].num_classes

    model = load_pretrained_model(num_labels=num_labels)

    params = model.params
    if args.bfloat16:
        params = jax.tree_map(lambda p: jnp.asarray(p, dtype=jnp.bfloat16), params)

    def loss_fn(params, batch, labels):
        outputs = model(**batch, params=params)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            jnp.asarray(outputs.logits, dtype=jnp.float32), labels
        ).mean()
        accuracy = (outputs.logits.argmax(axis=-1) == labels).mean()

        if args.optimise_accuracy:
            return -accuracy, loss
        return loss, accuracy

    if use_mezo:
        grad_loss_fn = mezo_value_and_grad(loss_fn, has_aux=True)
    else:
        grad_loss_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(params, batch, mezo_key):
        labels = batch.pop("label")
        if use_mezo:
            values, grad = grad_loss_fn(params, args.scale, mezo_key, batch, labels)
            if args.optimise_accuracy:
                accuracy, loss = values
                accuracy = -accuracy
            else:
                loss, accuracy = values
            scaled_grad = args.learning_rate * grad
            return loss, apply_updates(params, scaled_grad, mezo_key), accuracy
        else:
            (loss, accuracy), grad = grad_loss_fn(params, batch, labels)
            return loss, jax.tree_map(lambda p, u: p - args.learning_rate * u, params, grad), accuracy

    pb = tqdm(range(args.epochs * len(train_dataloader)))

    total_loss, total_accuracy = 0.0, 0.0
    steps = 0

    key, subkey = jax.random.split(key)
    for _ in range(args.epochs):
        for batch in train_dataloader:
            key, subkey = jax.random.split(key)
            batch = {k: v.numpy() for k, v in batch.items()}
            loss, params, accuracy = train_step(params, batch, subkey)

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            if steps % args.print_freq == 0:
                pb.set_description(
                    f"loss: {total_loss / args.print_freq} ~ accuracy: {100 * total_accuracy / args.print_freq}"
                )
                total_loss = 0.0
                total_accuracy = 0.0

            pb.update(1)
            steps += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0xFFFF)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--scale", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--disable_mezo", action="store_true")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--optimise_accuracy", action="store_true")
    args = parser.parse_args()
    main(args)
