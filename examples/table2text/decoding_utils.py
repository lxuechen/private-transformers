"""Utilities for generation."""
import logging
import sys
from typing import Optional

import tqdm
import transformers


def generate(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    loader=None,
    prompt_dataset=None,
    max_length=100,
    min_length=5,
    top_k=0,
    top_p=0.9,  # Only filter with top_p.
    repetition_penalty=1,
    do_sample=False,
    num_beams=5,
    bad_words_ids=None,
    dummy_token_id=-100,  # Used as mask.
    num_return_sequences=1,
    max_generations=sys.maxsize,
    device=None,
    padding_token="[PAD]",
    **kwargs,
):
    assert not model.training, "Generation must be when `model` is in eval mode."
    if kwargs:
        logging.warning(f"Unknown kwargs: {kwargs}")

    # These are linebreaks; generating these will mess up the evaluation, since those files assume one example per-line.
    if bad_words_ids is None:
        bad_words_ids = [[628], [198]]
        if padding_token in tokenizer.get_vocab():
            bad_words_ids.append(tokenizer.encode(padding_token))

    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_beams=num_beams,
        bad_words_ids=bad_words_ids,
        dummy_token_id=dummy_token_id,
        num_return_sequences=num_return_sequences,
        max_generations=max_generations,
        device=device,
        padding_token=padding_token,
    )
    if loader is not None:
        result = _generate_with_loader(loader=loader, **kwargs)
    elif prompt_dataset is not None:
        result = _generate_with_prompt_dataset(prompt_dataset=prompt_dataset, **kwargs)
    else:
        raise ValueError(f"`loader` and `prompt_dataset` cannot both be `None`.")

    return result


def _generate_with_loader(
    loader,

    model,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length,
    min_length,
    top_k,
    top_p,
    repetition_penalty,
    do_sample,
    num_beams,
    bad_words_ids,
    dummy_token_id,
    num_return_sequences,
    max_generations,
    device,
    padding_token,
):
    references = []
    full_generations = []  # Sentences including the prompt part.
    unstripped_generations = []
    generations = []

    stop_generation = False
    for batch_idx, batch in tqdm.tqdm(enumerate(loader), desc="generation"):
        if stop_generation:
            break

        batch_input_ids, batch_labels = batch["input_ids"], batch["labels"]
        # e.g., inputs_ids may be [[95, 123, 32], [198, 19, 120]], and
        # labels may be [[-100, 123, 32], [-100, -100, 120]

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            if stop_generation:
                break

            # Find the first pad token and end the sentence from there!
            if padding_token in tokenizer.get_vocab():
                pad_positions, = (
                    input_ids == tokenizer.encode(padding_token, return_tensors="pt").squeeze()
                ).nonzero(as_tuple=True)
                # Some sentences might have padding; others might not.
                if pad_positions.numel() == 0:
                    first_pad_position = None
                else:
                    first_pad_position = pad_positions[0]
                reference_str: str = tokenizer.decode(input_ids[:first_pad_position], clean_up_tokenization_spaces=True)
            else:
                reference_str: str = tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)
            references.append(reference_str)

            # Find the first non- -100 position. Note there are trailing -100s.
            non_prompt_positions, = (labels != dummy_token_id).nonzero(as_tuple=True)
            first_non_prompt_position = non_prompt_positions[0].item()
            prompt_len = first_non_prompt_position
            prompt_ids = input_ids[:prompt_len]

            output_ids = model.generate(
                input_ids=prompt_ids[None, ...].to(device),
                max_length=max_length + prompt_len,  # This cannot be a 0-D tensor!
                min_length=min_length,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                pad_token_id=tokenizer.eos_token_id,  # Stop the stupid logging...
            )
            output_ids = output_ids.squeeze(dim=0)  # Throw away batch dimension.

            whole_str: str = tokenizer.decode(output_ids, clean_up_tokenization_spaces=True)
            prompt_str: str = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=True)
            output_str: str = whole_str[len(prompt_str):]

            full_generations.append(whole_str)
            del whole_str, prompt_str

            # Remove potential eos_token at the end.
            eos_position: Optional[int] = output_str.find(tokenizer.eos_token)
            if eos_position == -1:  # Didn't generate eos_token; that's okay -- just skip!
                eos_position = None
            output_str = output_str[:eos_position]
            unstripped_generations.append(output_str)

            # Removing leading and trailing spaces.
            output_str = output_str.strip()

            generations.append(output_str)

            if len(generations) >= max_generations:
                stop_generation = True

    return full_generations, unstripped_generations, generations, references


def _generate_with_prompt_dataset(
    prompt_dataset,

    model,
    tokenizer,
    max_length,
    min_length,
    top_k,
    top_p,
    repetition_penalty,
    do_sample,
    num_beams,
    bad_words_ids,
    dummy_token_id,
    num_return_sequences,
    max_generations,
    device,
    padding_token,
):
    references = []
    full_generations = []  # Sentences including the prompt part.
    unstripped_generations = []
    generations = []

    stop_generation = False
    for input_ids in tqdm.tqdm(prompt_dataset, desc="generation"):
        if stop_generation:
            break

        prompt_len = len(input_ids[0])
        output_ids = model.generate(
            input_ids=input_ids.to(device),
            max_length=max_length + prompt_len,  # This cannot be a 0-D tensor!
            min_length=min_length,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            bad_words_ids=bad_words_ids,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            pad_token_id=tokenizer.eos_token_id,  # Stop the stupid logging...
        )
        output_ids = output_ids.squeeze(dim=0)  # Throw away batch dimension.
        input_ids = input_ids.squeeze(dim=0)

        whole_str: str = tokenizer.decode(output_ids, clean_up_tokenization_spaces=True)
        prompt_str: str = tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)
        output_str: str = whole_str[len(prompt_str):]

        full_generations.append(whole_str)
        del whole_str, prompt_str

        # Remove potential eos_token at the end.
        eos_position: Optional[int] = output_str.find(tokenizer.eos_token)
        if eos_position == -1:  # Didn't generate eos_token; that's okay -- just skip!
            eos_position = None
        output_str = output_str[:eos_position]
        unstripped_generations.append(output_str)

        # Removing leading and trailing spaces.
        output_str = output_str.strip()

        generations.append(output_str)

        if len(generations) >= max_generations:
            stop_generation = True
    return full_generations, unstripped_generations, generations, references
