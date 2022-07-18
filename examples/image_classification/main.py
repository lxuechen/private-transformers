"""CIFAR-10 classification with Vi-T."""
import logging

import fire
import torch
import torch.nn.functional as F
import tqdm
import transformers
from ml_swissknife import utils
from torchvision import transforms

import private_transformers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(loader, model):
    model.eval()
    xents, zeons = [], []
    for i, (images, labels) in enumerate(loader):
        images, labels = tuple(t.to(device) for t in (images, labels))
        logits = model(pixel_values=images).logits
        xents.append(F.cross_entropy(logits, labels, reduction='none'))
        zeons.append(logits.argmax(dim=-1).ne(labels).float())
    return tuple(torch.cat(lst).mean().item() for lst in (xents, zeons))


def main(
    model_name_or_path='google/vit-base-patch16-224',
    train_batch_size=1000,
    per_device_train_batch_size=50,
    test_batch_size=500,
    epochs=10,
    target_epsilon=2,
    lr=2e-3,
    max_grad_norm=0.1,
    linear_probe=True,
):
    gradient_accumulation_steps = train_batch_size // per_device_train_batch_size

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader, test_loader = utils.get_loader(
        data_name='cifar10',
        task="classification",
        train_batch_size=per_device_train_batch_size,
        test_batch_size=test_batch_size,
        data_aug=False,
        train_transform=image_transform,
        test_transform=image_transform,
    )

    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = 10
    model = transformers.ViTForImageClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
    ).to(device)
    if linear_probe:
        model.requires_grad_(False)
        model.classifier.requires_grad_(True)
        logging.warning("Linear probe classification head.")
    else:
        private_transformers.freeze_isolated_params_for_vit(model)
        logging.warning("Full fine-tune up to isolated embedding parameters.")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    privacy_engine = private_transformers.PrivacyEngine(
        model,
        batch_size=train_batch_size,
        sample_size=50000,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        target_epsilon=target_epsilon,
    )
    privacy_engine.attach(optimizer)

    train_loss_meter = utils.AvgMeter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pbar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
        for global_step, (images, labels) in pbar:
            model.train()
            images, labels = tuple(t.to(device) for t in (images, labels))
            logits = model(pixel_values=images).logits
            loss = F.cross_entropy(logits, labels, reduction="none")
            train_loss_meter.step(loss.mean().item())
            if global_step % gradient_accumulation_steps == 0:
                optimizer.step(loss=loss)
                optimizer.zero_grad()
            else:
                optimizer.virtual_step(loss=loss)
            pbar.set_description(f"Train loss running average: {train_loss_meter.item():.4f}")
        avg_xent, avg_zeon = evaluate(test_loader, model)
        logging.warning(
            f"Epoch: {epoch}, average cross ent loss: {avg_xent:.4f}, average zero one loss: {avg_zeon:.4f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
