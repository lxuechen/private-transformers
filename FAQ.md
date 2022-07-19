## FAQ

### How do I perform gradient accumulation?

Use `virtual_step` in combination with `step`. For example, the following gives a simplified demo of the structure:

```python
import torch, transformers, private_transformers

gradient_accumulation_steps = 10  # Take an update once this many iterations.

batches = ...  # Data.
model = transformers.AutoModelWithLMHead.from_pretrained('gpt')
optimizer = torch.optim.Adam(model.parameters())
privacy_engine = private_transformers.PrivacyEngine(...)
privacy_engine.attach(optimizer)

for i, batch in enumerate(batches, 1):
    loss = model(batch)
    if i % gradient_accumulation_steps == 0:
        optimizer.step(loss=loss)
        optimizer.zero_grad()
    else:
        optimizer.virtual_step(loss=loss)
```

### What is ghost clipping?

It's a per example gradient clipping (then summing) technique that avoids instantiating per example gradients. It can
make private training have almost the same memory cost as non-private training.
The method is based on accumulating gradient norms on a layer-by-layer basis first demonstrated
in [this work](https://arxiv.org/abs/2009.03106).
We implemented and extended this method so that computing gradient norms for linear layers can be cheap; this is based on a
linear algebra identity that we derived in [this work](https://arxiv.org/pdf/2110.05679.pdf).
[Subsequent work](https://arxiv.org/abs/2205.10683) adapted the overall approach to suit training convolutional layers.

### How did you test that ghost clipping gives the 'right' gradients?

We ran stringent numerical tests to ensure the double-backward implementation is correct (e.g., remove sources of
randomness like dropout and compare gradients from double backward against gradients from autodiff + for loop).
Check out files in the `tests` folder for more on this.

### When can't I use ghost clipping?

Ghost clipping can't handle parameter sharing, that's why in our code, we separate the lm-head out from the embedding
layer for generation tasks. Similarly, it can't be applied to fine-tuning ALBERT which ties weights across many layers
of the model.

### What if I want to freeze some parameters of the network while updating all others?

Before creating the privacy engine and optimizer, set parts of the model which won't be optimized to
have `.requires_grad=False`. The privacy engine will do the rest for you. For instance:

```python
import transformers, private_transformers

model = transformers.AutoModelWithLMHead.from_pretrained('gpt')
# Input embeddings aren't optimized; this line needs to proceed privacy engine creation.
model.get_input_embeddings().requires_grad_(False)
privacy_engine = private_transformers.PrivacyEngine(model, ...)
```