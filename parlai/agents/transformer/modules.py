#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements NN code for transformers.

Original paper: https://arxiv.org/abs/1706.03762. (Vaswani, 2017). The
`Annotated Transformer` (Rush, 2018) is an excellent reading guide which explains
much of the mechanics of the Transformer model
(http://nlp.seas.harvard.edu/2018/04/03/attention.html).

This module also supports special segments (ala BERT;
https://arxiv.org/abs/1810.04805), and a few different variations seen in the
literature (BERT and XLM; https://arxiv.org/abs/1901.07291).
"""

import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union, Set

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

    APEX_LAYER_NORM = True
except ImportError:
    from torch.nn import LayerNorm

    APEX_LAYER_NORM = False

LAYER_NORM_EPS = 1e-5  # Epsilon for layer norm.
"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

_seen_logs: Set[str] = set()


def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def warn_once(msg: str) -> None:
    """
    Log a warning, but only once.

    :param str msg: Message to display
    """
    global _seen_logs
    if msg not in _seen_logs:
        _seen_logs.add(msg)
        logging.warning(msg)


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm.
    """
    is_cpu = tensor.device == 'cpu' or tensor.device.type == 'cpu'
    if APEX_LAYER_NORM and not is_cpu:
        # fused_layer_norm has a bug around multi-device networks.
        # https://github.com/NVIDIA/apex/issues/770
        # https://github.com/NVIDIA/apex/issues/371
        with torch.cuda.device(tensor.device):
            return norm_layer(tensor)
    else:
        return norm_layer(tensor)


def _create_embeddings(tokenizer, embedding_size, padding_idx):
    """
    Create and initialize word embeddings.
    """
    e = nn.Embedding(tokenizer.vocab_size, embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


def create_position_codes(n_pos, dim, out):
    """
    Create positional codes and store them in ``out``.
    """
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ]
    )

    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)


def get_head_mask(head_mask, batch_size, n_layers):
    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(n_layers, batch_size, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(-1, batch_size, -1, -1, -1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        n_heads = head_mask.size()[2]
        head_mask = head_mask.reshape(-1, batch_size*n_heads, 1, 1)
    return head_mask


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions:
        Size of the position embeddings matrix.
    :param int n_segments:
        Number of segments/lang/sentence embeddings.
    :param activation:
        Type of nonlinear activation. Can be relu or gelu.
    :param variant:
        Which transformer architecture to use. Could be AIAYN or XLM.
        Future versions may support things like GPT-2, ...
    :param output_scaling:
        Scale the outputs by a given scalar
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        attention_head_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction_type='mean',
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout_frac = dropout
        self.dropout = nn.Dropout(p=self.dropout_frac)
        self.variant = variant
        self.n_segments = n_segments

        self.n_positions = n_positions
        self.out_dim = embedding_size
        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # embedding normalization
        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_head_size=attention_head_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    variant=variant,
                    activation=activation,
                )
            )
        self.output_scaling = output_scaling

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.

        :return (tensor, mask):
            return embedded input and mask
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros_like(input)  # type: ignore
            tensor = tensor + self.segment_embeddings(segments)

        return tensor, mask

    def forward_layers(
        self,
        tensor: torch.Tensor,
        mask: torch.BoolTensor,
        head_mask: Optional[torch.LongTensor]=None,
    ) -> torch.Tensor:
        """
        Apply transformer layers to input.

        :param tensor:
            embedded input
        :param mask:
            mask of input

        :return tensor:
            return embedding after applying transformer layers
        """
        batch_size = tensor.size()[0]
        head_mask = get_head_mask(head_mask, batch_size, self.n_layers)

        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask,
                                    head_mask=head_mask[i] if head_mask is not None else head_mask)

        return tensor

    def reduce_output(
        self, tensor: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        """
        Reduce transformer output at end of forward pass.

        :param tensor:
            encoded input
        :param mask:
            mask for encoded input

        :return (tensor, mask):
            returns the reduced tensor, and mask if appropriate
        """
        tensor *= self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :], None
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0], None
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output, None
        elif self.reduction_type is None or 'none' in self.reduction_type:
            return tensor, mask
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )

    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        # embed input
        tensor, mask = self.forward_embedding(input, positions, segments)

        if self.variant == 'xlm' or self.variant == 'bart':
            tensor = _normalize(tensor, self.norm_embeddings)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)

        # apply transformer layers
        tensor = self.forward_layers(tensor, mask, head_mask=head_mask)

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # reduce output
        tensor, out_mask = self.reduce_output(tensor, mask)
        if out_mask is not None:
            return tensor, out_mask
        else:
            return tensor


class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer.
    """

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_head_size=-1,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant=None,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.attention = MultiHeadAttention(
            n_heads, embedding_size,
            attention_head_size=attention_head_size,
            dropout=attention_dropout  # --attention-dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.ffn = TransformerFFN(
            embedding_size,
            ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask, head_mask=None):
        """
        Forward pass.
        """
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm1)
        attended_tensor, _ = self.attention(tensor, mask=mask, head_mask=head_mask)
        tensor = residual + self.dropout(attended_tensor)
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = _normalize(tensor, self.norm1)
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm2)
        tensor = residual + self.dropout(self.ffn(tensor))
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = _normalize(tensor, self.norm2)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        attention_head_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
        variant='aiayn',
        activation='relu',
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.activation = activation
        self.variant = variant

        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.n_positions = n_positions
        self.out_dim = embedding_size
        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
            if self.variant == 'xlm':
                warn_once(
                    'DEPRECATED: XLM should only be used for backwards compatibility, '
                    'as it involves a less-stable layernorm operation.'
                )
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_head_size=attention_head_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    activation=activation,
                    variant=variant,
                )
            )

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ):
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch, seqlen] input:
            The target input IDs
        :param LongTensor[batch, seqlen] positions:
            Positions for input IDs. If None, computes defaults.
        :param LongTensor[batch, seqlen] segements:
            Segment IDs for extra embedding features. If None, not used.

        :return (tensor, mask):
            embeded input and mask
        """
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if self.variant == 'bart':
            tensor = _normalize(tensor, self.norm_embeddings)

        return tensor

    def forward_layers(
        self,
        tensor: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: Dict[int, torch.Tensor],
        self_head_mask: Optional[torch.Tensor] = None,
        cross_head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param enc_out:
            encoder outputs
        :param enc_mask:
            encoder output mask
        :param incr_state:
            Dict mapping layer_idx to incremental state

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}

        batch_size = tensor.size()[0]
        self_head_mask = get_head_mask(self_head_mask, batch_size, self.n_layers)
        cross_head_mask = get_head_mask(cross_head_mask, batch_size, self.n_layers)

        for idx, layer in enumerate(self.layers):
            tensor, new_incr_state[idx] = layer(
                x=tensor,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
                incr_state=incr_state.get(idx),
                self_head_mask=self_head_mask[idx] if self_head_mask is not None else self_head_mask,
                cross_head_mask=cross_head_mask[idx] if cross_head_mask is not None else cross_head_mask
            )

        return tensor, new_incr_state

    def forward(self, input, encoder_state, incr_state=None, self_head_mask=None, cross_head_mask=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.forward_embedding(input, positions)

        tensor = self.dropout(tensor)  # --dropout

        tensor, new_incr_state = self.forward_layers(
            tensor, encoder_output, encoder_mask, incr_state,
            self_head_mask=self_head_mask, cross_head_mask=cross_head_mask
        )

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)

        return tensor, new_incr_state


class TransformerDecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a casaul (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_head_size=-1,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant='aiayn',
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size,
            attention_head_size=attention_head_size,
            dropout=attention_dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size,
            attention_head_size=attention_head_size,
            dropout=attention_dropout
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = TransformerFFN(
            embedding_size, ffn_size, relu_dropout=relu_dropout, activation=activation
        )
        self.norm3 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(self, x, encoder_output, encoder_mask, incr_state=None, self_head_mask=None, cross_head_mask=None):
        """
        Forward pass.

        The incremental state is a dict with values for self- and encoder-attention
        states.
        """

        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm1)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            head_mask=self_head_mask,
        )
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm1)

        residual = x
        # encoder_attn_layer_norm norm 2
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm2)
        x, final_encoder_attn_incr_state = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
            incr_state=incr_state.get('encoder_attn'),
            static_kv=True,
            head_mask=cross_head_mask,
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm3)
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm3)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_attn': final_encoder_attn_incr_state,
        }
        return x, new_incr_state

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

    def reorder_incremental_state(
        self, incremental_state: Dict[str, dict], inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {
            'self_attn': self.self_attention,
            'encoder_attn': self.encoder_attention,
        }
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }


class TorchGeneratorModel(nn.Module, ABC):
    """
    Abstract TorchGeneratorModel.

    This interface expects you to implement model with the following reqs:

    :attribute model.encoder:
        takes input returns tuple (enc_out, enc_hidden, attn_mask)

    :attribute model.decoder:
        takes decoder params and returns decoder outputs after attn

    :attribute model.output:
        takes decoder outputs and returns distr over dictionary
    """

    def __init__(
        self,
        padding_idx=0,
        start_idx=1,
        end_idx=2,
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
    ):
        super().__init__()
        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.START_IDX = start_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        return torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)

    def decode_forced(self, encoder_states, ys, self_head_mask=None, cross_head_mask=None):
        """
        Decode with a fixed, true sequence, computing loss.

        Useful for training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(bsz, inputs)
        latent, _ = self.decoder(inputs, encoder_states,
                                 self_head_mask=self_head_mask,
                                 cross_head_mask=cross_head_mask)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    @abstractmethod
    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        pass

    @abstractmethod
    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        pass

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None,
                encoder_head_mask=None, decoder_self_head_mask=None, decoder_cross_head_mask=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs, head_mask=encoder_head_mask)

        # use teacher forcing
        scores, preds = self.decode_forced(encoder_states, ys,
                                           self_head_mask=decoder_self_head_mask,
                                           cross_head_mask=decoder_cross_head_mask)
        return scores, preds, encoder_states


class TransformerGeneratorModel(TorchGeneratorModel):
    """
    Implements a full generator model, with one encoder and one decoder.
    """

    @classmethod
    def build_encoder(
        cls,
        opt,
        tokenizer,
        embedding=None,
        padding_idx=None,
        reduction_type='mean',
        n_positions=1024,
        n_segments=0,
    ):
        n_layers = (
            opt['n_encoder_layers']
            if opt.get('n_encoder_layers', -1) > 0
            else opt['n_layers']
        )
        return TransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=n_layers,
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            attention_head_size=opt['attention_head_size'],
            vocabulary_size=tokenizer.vocab_size,
            embedding=embedding,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=padding_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            reduction_type=reduction_type,
            n_positions=n_positions,
            n_segments=n_segments,
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )

    @classmethod
    def build_decoder(
        cls,
        opt,
        tokenizer,
        embedding=None,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
    ):
        n_layers = (
            opt['n_decoder_layers']
            if opt.get('n_decoder_layers', -1) > 0
            else opt['n_layers']
        )
        return TransformerDecoder(
            n_heads=opt['n_heads'],
            n_layers=n_layers,
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            attention_head_size=opt['attention_head_size'],
            vocabulary_size=tokenizer.vocab_size,
            embedding=embedding,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=padding_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            n_positions=n_positions,
            activation=opt['activation'],
            variant=opt['variant'],
            n_segments=n_segments,
        )

    def __init__(self, opt, tokenizer):
        self.pad_idx = tokenizer.pad_token_id
        self.start_idx = tokenizer.bos_token_id
        self.end_idx = tokenizer.eos_token_id
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.embeddings = _create_embeddings(
            tokenizer, opt['embedding_size'], self.pad_idx
        )

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0,
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024
        n_segments = opt.get('n_segments', 0)

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = self.build_encoder(
            opt,
            tokenizer,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            n_positions=n_positions,
            n_segments=n_segments,
        )
        self.decoder = self.build_decoder(
            opt, tokenizer, self.embeddings, self.pad_idx, n_positions=n_positions
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)
        return output


class MultiHeadAttention(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Transformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(self, n_heads, dim, attention_head_size=-1, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        if attention_head_size > 0:
            self.dim_per_head = attention_head_size
            self.all_head_size = attention_head_size * self.n_heads
        else:
            self.dim_per_head = dim // self.n_heads
            self.all_head_size = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, self.all_head_size)
        self.k_lin = nn.Linear(dim, self.all_head_size)
        self.v_lin = nn.Linear(dim, self.all_head_size)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(self.all_head_size, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(  # type: ignore
        # TODO: remove type ignore with pytorch 1.5:
        # https://github.com/pytorch/pytorch/pull/31057
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        static_kv: bool = False,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        :param query: attention query
        :param key: attention key
        :param value: attention value
        :param mask: tensor in which True means that we are allowing attention and False
          means we are blocking it. Mask is:
          - [B, key_len] (encoder self-attn and decoder enc/dec attn)
          - [B, query_len, key_len] (decoder self-attn)
          - [B, 1, 1] (decoder self-attn with incr_state caching)
        :param incr_state: dictionary with values representing the previous states of
          the key, value, and mask
        :param static_kv: True if the key and value are held constant during decoding
          (as in encoder/decoder attention)
        :return: (final attended tensor, new incremental state)
        """

        batch_size, query_len, dim = query.size()
        assert (
            dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = self.dim_per_head
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
            _, _key_len, dim = query.size()
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key

        assert key is not None  # let mypy know we sorted this
        _, _key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        # Prepend incremental states. For each of the key, value, and mask, see if
        # a previous incremental state exists, and if so, reshape it to match the shape
        # of the new state. Concatenate the previous and new states to match what the
        # full state would have been if we had not cached. (If we are using static_kv,
        # these three states are unchanging, so just re-use the cached states.)
        if incr_state is None:
            incr_state = {}
        if 'prev_key' in incr_state:
            prev_key = incr_state['prev_key'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                k = prev_key
            else:
                k = torch.cat([prev_key, k], dim=1)
        if 'prev_value' in incr_state:
            prev_value = incr_state['prev_value'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                v = prev_value
            else:
                v = torch.cat([prev_value, v], dim=1)
        if 'prev_mask' in incr_state:
            if static_kv:
                mask = incr_state['prev_mask']
            else:
                mask = torch.cat([incr_state['prev_mask'], mask], dim=2)
                # Prepend along the key_len dimension (analogous to
                # incr_state['prev_key'])

        # Save new incremental states. We reshape to allow for reordering along batch
        # dimension.
        new_incr_state = {
            'prev_key': k.view(batch_size, n_heads, -1, dim_per_head),
            'prev_value': v.view(batch_size, n_heads, -1, dim_per_head),
            'prev_mask': mask,
        }

        full_key_len = k.size(1)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, full_key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, full_key_len)
            .view(batch_size * n_heads, query_len, full_key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(
            dot_prod, dim=-1, dtype=torch.float  # type: ignore
        ).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        # head mask here
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, self.all_head_size)
        )

        out = self.out_lin(attentioned)

        return out, new_incr_state

    def reorder_incremental_state(
        self, incremental_state: Dict[str, torch.Tensor], inds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Reorder the input incremental-state tensors.
        """
        return {
            key: torch.index_select(val, 0, inds.to(val.device)).contiguous()
            for key, val in incremental_state.items()
        }


class TransformerFFN(nn.Module):
    """
    Implements the FFN part of the transformer.
    """

    def __init__(self, dim, dim_hidden, relu_dropout=0, activation='relu'):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = F.gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        """
        Forward pass.
        """
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x
