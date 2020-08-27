#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transformer Agents.
"""
from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.distill.torch_generator_agent import TorchDistillGeneratorAgent
from parlai.utils.misc import recursive_getattr
from parlai.utils.logging import logging

from parlai.agents.transformer.modules import (
    TransformerGeneratorModel,
)

import torch
import parlai.chat_service.utils.config as config_utils

def add_common_cmdline_args(argparser):
    """
    Add common command line args.
    """
    argparser.add_argument(
        '--student-config',
        type=dict,
        help="path to configuration yml file which will contain all the infos of decoder and encoder"
    )
    argparser.add_argument(
        '--teacher-config',
        type=dict,
        help="path to configuration yml file which will contain all the infos of decoder and encoder"
    )
    argparser.add_argument(
        '-esz',
        '--embedding-size',
        type=int,
        default=300,
        help='Size of all embedding layers',
    )
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument(
        '-hid',
        '--ffn-size',
        type=int,
        default=300,
        help='Hidden size of the FFN layers',
    )
    argparser.add_argument(
        '--dropout', type=float, default=0.0, help='Dropout used in Vaswani 2017.'
    )
    argparser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax.',
    )
    argparser.add_argument(
        '--relu-dropout',
        type=float,
        default=0.0,
        help='Dropout used after ReLU. From tensor2tensor.',
    )
    argparser.add_argument(
        '--n-heads', type=int, default=2, help='Number of multihead attention heads'
    )
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument(
        '--n-positions',
        type=int,
        default=None,
        hidden=True,
        help='Number of positional embeddings to learn. Defaults '
        'to truncate or 1024 if not provided.',
    )
    argparser.add_argument(
        '--n-segments',
        type=int,
        default=0,
        help='The number of segments that support the model. '
        'If zero no segment and no langs_embedding.',
    )
    argparser.add_argument(
        '--variant',
        choices={'aiayn', 'xlm', 'prelayernorm', 'bart'},
        default='aiayn',
        help='Chooses locations of layer norms, etc. prelayernorm '
        'is used to match some fairseq models',
        recommended='xlm',
    )
    argparser.add_argument(
        '--activation',
        choices={'relu', 'gelu'},
        default='relu',
        help='Nonlinear activation to use. AIAYN uses relu, but '
        'more recent papers prefer gelu.',
        recommended='gelu',
    )
    argparser.add_argument(
        '--output-scaling',
        type=float,
        default=1.0,
        help='scale the output of every transformer by this quantity.',
    )
    argparser.add_argument(
        '--share-word-embeddings',
        type='bool',
        default=True,
        help='Share word embeddings table for candidate and context'
        'in the memory network',
    )
    argparser.add_argument(
        '-nel',
        '--n-encoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    argparser.add_argument(
        '-ndl',
        '--n-decoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    argparser.add_argument(
        '--model-parallel',
        type='bool',
        default=False,
        help='Shard the layers across multiple GPUs.',
    )

class Transformer(Agent):
    """
    Placeholder Transformer Agent.

    Placeholder class, which just throws an error telling the user to specify whether
    they want the ranker or the generator.
    """

    def __init__(self, opt, shared=None):
        raise RuntimeError(
            "`--model transformer` is not a valid choice. Please select either "
            "`--model transformer/ranker` or `--model transformer/generator"
        )

class TransformerGeneratorAgent(TorchDistillGeneratorAgent):
    """
    TransformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        student_model = TransformerGeneratorAgent(self.opt['student_config'], self.dict)
        model = TransformerGeneratorModel(self.opt['teacher_config'], self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                student_model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        
        return student_model, model

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when are adding extra special tokens.
        """
        # map extra special tokens carefully
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in [
            'embeddings.weight',
            'encoder.embeddings.weight',
            'decoder.embeddings.weight',
        ]:
            # get new_embs
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[emb_weights] = new_embs

        return state_dict
