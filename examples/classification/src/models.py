"""Custom models for few-shot learning specific operations."""

import logging

import torch
import torch.nn as nn
from transformers.activations import gelu
from transformers.models.albert.modeling_albert import AlbertModel, AlbertMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead, BertForMaskedLM
from transformers.models.distilbert import DistilBertModel, DistilBertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLMHead

logger = logging.getLogger(__name__)


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[
        :old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertForMaskedLM):

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=add_pooling_layer)
        # lxuechen: The name of this variable must be `.cls`! Otherwise, error in loading
        # and you implicitly get random weights!!!
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, labels=None):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Get <mask> token representation
        sequence_output, = outputs[:1]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class DistilBertForPromptFinetuning(DistilBertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.lb = None
        self.ub = None
        self.return_full_softmax = None

    def get_input_embeddings(self):
        return self.distilbert.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, mask_pos=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output, = outputs[:1]

        batch_size = input_ids.size(0)
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()  # (batch_size,), a sequence of ints.
        sequence_mask_output = sequence_output[torch.arange(batch_size), mask_pos]

        prediction_logits = self.vocab_transform(sequence_mask_output)  # (bs, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, dim)
        prediction_mask_scores = self.vocab_projector(prediction_logits)  # (bs, vocab_size)

        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=add_pooling_layer)
        # lxuechen: The name of this variable must be `.lm_head`! Otherwise, error in loading,
        #   and you implicitly get random weights!!!
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len).
            attention_mask: (batch_size, seq_len).
            mask_pos: (batch_size, 1).
            labels: (batch_size,).

        Returns:
            tuple of logits (and maybe loss).
        """
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()  # (batch_size,), a sequence of ints.

        # Encode everything
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # --- lxuechen: pooled_output does not seem to be used!
        sequence_output, = outputs[:1]
        # Pick the entries that correspond to labels.
        sequence_mask_output = sequence_output[torch.arange(batch_size), mask_pos]
        # ---

        # # Get <mask> token representation
        # sequence_output, pooled_output = outputs[:2]
        # sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class AlbertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.albert = AlbertModel(config, add_pooling_layer=add_pooling_layer)
        # lxuechen: The name of this variable must be `.predictions`! Otherwise, error in loading
        # and you implicitly get random weights!!!
        self.predictions = AlbertMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def get_input_embeddings(self):
        return self.albert.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len).
            token_type_ids: (batch_size, seq_len).
            attention_mask: (batch_size, seq_len).
            mask_pos: (batch_size, 1).
            labels: (batch_size,).
        Returns:
            tuple/dict of loss and logits.
        """
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()  # (batch_size,), a sequence of ints.

        # Encode everything
        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # --- lxuechen: pooled_output does not seem to be used!
        sequence_output, = outputs[:1]
        # Pick the entries that correspond to labels.
        sequence_mask_output = sequence_output[torch.arange(batch_size), mask_pos]
        # ---

        # # Get <mask> token representation
        # sequence_output, pooled_output = outputs[:2]
        # sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.predictions(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        # --- lxuechen: This is only slow for large number of labels.
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output
