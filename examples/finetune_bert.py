from typing import Optional

import datasets
import jax
import optax
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

from mezo_jax import apply_updates, mezo_value_and_grad

# Mostly following this tutorial https://huggingface.co/docs/transformers/training


def load_tokenizer(name: str = "bert-base-cased"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def load_dataset(tokenizer: AutoTokenizer, name: str = "yelp_review_full", batch_size: int = 16, num_workers: int = 4):
    dataset = datasets.load_dataset(name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    dataset.set_format("torch")

    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=batch_size, num_workers=num_workers)
    eval_dataloader = DataLoader(dataset["test"], shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, eval_dataloader


def load_pretrained_model(name: str = "bert-base-cased", num_labels: int = 5):
    return FlaxAutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)


def main(args):
    seed = 0xFFFF
    lr = 1e-5
    scale = 1e-3
    epochs = 10
    batch_size = 64
    num_workers = 8

    key = jax.random.PRNGKey(seed)
    tokenizer = load_tokenizer()
    train_dataloader, eval_dataloader = load_dataset(tokenizer, batch_size=batch_size, num_workers=num_workers)
    model = load_pretrained_model()

    params = model.params

    def loss_fn(params, batch, labels):
        outputs = model(**batch, params=params)
        loss = optax.softmax_cross_entropy_with_integer_labels(outputs.logits, labels).mean()

        return loss

    grad_loss_fn = mezo_value_and_grad(loss_fn)

    @jax.jit
    def train_step(params, batch, mezo_key):
        labels = batch.pop("label")
        loss, grad = grad_loss_fn(params, scale, mezo_key, batch, labels)
        scaled_grad = lr * grad
        return loss, apply_updates(params, scaled_grad, mezo_key)

    pb = tqdm(range(epochs * len(train_dataloader)))

    total_loss = 0.0
    steps = 0
    freq = 10
    for _ in range(epochs):
        for batch in train_dataloader:
            key, subkey = jax.random.split(key)
            batch = {k: v.numpy() for k, v in batch.items()}
            loss, params = train_step(params, batch, subkey)

            total_loss += loss
            if steps % freq == 0:
                pb.set_description(f"loss: {total_loss / freq}")
                total_loss = 0.0

            pb.update(1)
            steps += 1


if __name__ == "__main__":
    main(None)
