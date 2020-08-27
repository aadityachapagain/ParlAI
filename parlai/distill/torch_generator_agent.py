from parlai.core.torch_generator_agent import (
    TorchGeneratorAgent,
    TorchGeneratorModel,
    PPLMetric,
    TreeSearch,
    _HypothesisTail,
    GreedySearch,
    BeamSearch,
    total_parameters,
    trainable_parameters,
    DelayedBeamSearch,
    NucleusSampling,
    SumMetric,
    SearchBlocklist
)
from parlai.core.torch_agent import TorchAgent, Batch, Output, DictionaryAgent
from parlai.core.torch_agent import Batch, Output, DictionaryAgent
from parlai.utils.distributed import is_distributed, sync_parameters

from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import math
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Union, List, Tuple, Optional
import random
import os
import parlai.utils.logging as logging
from torch import optim

from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.utils.thread import SharedTable
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
    fp16_apex_available,
    fp16_optimizer_wrapper,
    MemoryEfficientFP16Optimizer,
    MemoryEfficientFP16Adam,
    Adafactor,
)
from parlai.core.metrics import (
    Metrics,
    Metric,
    GlobalAverageMetric,
    GlobalFixedMetric,
    GlobalTimerMetric,
)
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor, atomic_save

from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim

from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.utils.thread import SharedTable
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message

try:
    from nltk.translate import bleu_score as nltkbleu

except ImportError:
    nltkbleu = None

try:
    from fairseq import bleu as fairseq_bleu

except ImportError:
    fairseq_bleu = None

class TorchDistillGeneratorAgent(TorchGeneratorAgent):
    def __init__(self, opt: Opt, shared=None):
        """
        Initialize agent.
        """

        init_student_model, is_finetune = self._get_student_init_model(opt, shared)
        init_model = self._get_init_model(opt, shared)
        super().__init__(opt, shared)
        opt = self.opt

        # Safety checkers to ensure TorchAgent assumptions aren't being violated.
        self.__expecting_clear_history = False
        self.__expecting_to_reply = False

        # used for sharing metrics back to the teacher
        self._local_metrics: Dict[str, List[Metric]] = {}
        # we may want to temporarily disable local metrics, roughly similar to
        # `with torch.no_grad`. See TorchGeneratorAgent._init_cuda_buffer for
        # example
        self.__local_metrics_enabled = True

        # check for cuda
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        if self.use_cuda:
            if not shared:
                logging.info('Using CUDA')
            if not shared and opt['gpu'] != -1:
                torch.cuda.set_device(opt['gpu'])

        # whether we're using multi-gpu, a few different ways. these are not
        # supported by all models, but we can still keep track of the options
        self.model_parallel = opt.get('model_parallel', False) and self.use_cuda
        self.data_parallel = opt.get('data_parallel', False) and self.use_cuda
        if self.data_parallel and is_distributed():
            raise RuntimeError('Cannot combine --data-parallel and distributed mode.')
        if self.model_parallel and self.data_parallel:
            raise RuntimeError('Cannot combine --data-parallel and --model-parallel.')

        # indicate whether using fp16
        self.fp16 = self.use_cuda and self.opt.get('fp16', False)
        if self.fp16:
            # check that the implementation requested is available
            self.fp16_impl = self.opt.get('fp16_impl', 'apex')
            if self.fp16_impl == 'apex' and not fp16_apex_available():
                self.fp16 = False

        if shared is None:
            # intitialize any important structures from scratch
            self.dict = self.build_dictionary()

            if opt.get('fp16') or opt.get('force_fp16_tokens'):
                # Volta cores revert to FP32 hardware if tensors are not multiples
                # of 8 in all dimensions. This INCLUDES the embeddings layer! As
                # such, we need some extra magic to ensure the dictionary is padded
                # with extra tokens to make it a multiple of 8.
                from parlai.utils.torch import FP16_PAD_SIZE

                if len(self.dict) % FP16_PAD_SIZE != 0:
                    for i in range(FP16_PAD_SIZE - len(self.dict) % FP16_PAD_SIZE):
                        self.dict['__FP16_PAD_{}__'.format(i)] = 1

            # global_metrics keeps track of batch-level or global-level metrics
            self.global_metrics = Metrics(opt.get('numthreads', 1) > 1, shared=None)
            # self.metrics is there for legacy reasons
            self.metrics: Dict[str, Any] = {}
        else:
            # copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']
            self.model = shared['model']
            self.student_model = shared['student_model']
            self.criterion = shared['criterion']
            self.metrics = shared['metrics']
            self.global_metrics = Metrics(
                opt.get('numthreads', 1) > 1, shared=shared['global_metrics']
            )

        if opt.get('numthreads', 1) > 1:
            torch.set_num_threads(1)

        # Default to the class name, sans "Agent". child can override
        self.id = type(self).__name__.replace("Agent", "")

        # now set up any fields that all instances may need
        self.EMPTY = torch.zeros(0, dtype=torch.long)
        self.NULL_IDX = self.dict[self.dict.null_token]
        self.START_IDX = self.dict[self.dict.start_token]
        self.END_IDX = self.dict[self.dict.end_token]

        # for gradient acumulation
        self._number_grad_accum = 0
        # for the LR scheduler
        self._number_training_updates = 0
        # fixed random seed
        self.random = random.Random(42)
        # can remember as few as zero utterances if desired
        self.histsz = opt['history_size']
        # truncate == 0 might give funny behavior
        self.truncate = opt['truncate'] if opt['truncate'] >= 0 else None
        text_truncate = opt.get('text_truncate') or opt['truncate']
        self.text_truncate = text_truncate if text_truncate >= 0 else None
        label_truncate = opt.get('label_truncate') or opt['truncate']
        self.label_truncate = label_truncate if label_truncate >= 0 else None
        # stores up to hist_utt past observations within current dialog
        self.history = self.build_history()
        self.history_reversed = opt.get('history_reversed', False)

        self.is_training = False  # track whether model is training
        self.rank_candidates = opt['rank_candidates']
        self.add_person_tokens = opt.get('person_tokens', False)
        # set interactive mode or not according to options.
        self.set_interactive_mode(opt.get('interactive_mode', False), shared)

        self.beam_size = opt.get('beam_size', 1)
        self.beam_min_length = opt.get('beam_min_length', 1)
        self.beam_block_ngram = opt.get('beam_block_ngram', -1)
        self.beam_context_block_ngram = opt.get('beam_context_block_ngram', -1)
        self.beam_block_full_context = opt.get('beam_block_full_context', False)
        self.temperature = opt.get('temperature', 1.0)
        assert self.temperature > 0, '--temperature must be greater than 0'
        self.output_token_losses = opt.get('verbose', False)
        self.compute_tokenized_bleu = opt.get('compute_tokenized_bleu', False)
        self.beam_block_list: Optional[SearchBlocklist] = None

        if shared:
            # set up shared properties
            states = shared.get('states', {})
            self.beam_block_list = shared.get('beam_block_list')
        else:
            # this is not a shared instance of this class, so do full init
            self.criterion = self.build_criterion()
            # ensure all distributed copies will always be in sync
            self.student_model, self.model = self.build_model()

            # load the block_list for beam search
            self.beam_block_list = self._load_beam_block_list()

            if self.model is None or self.criterion is None:
                raise AttributeError(
                    'build_model() and build_criterion() need to return the model or criterion'
                )
            if self.use_cuda:
                if self.model_parallel:
                    self.model = PipelineHelper().make_parallel(self.model)
                    self.student_model = PipelineHelper().make_parallel(self.student_model)
                else:
                    self.model.cuda()
                    self.student_model.cuda()
                self.criterion.cuda()

            sync_parameters(self.model)
            sync_parameters(self.student_model)
            train_params = trainable_parameters(self.student_model)
            total_params = total_parameters(self.student_model)
            logging.info(
                f"Total parameters: {total_params:,d} ({train_params:,d} trainable)"
            )

            if self.fp16:
                self.model = self.model.half()
                self.student_model = self.student_model.half()

            if init_model is not None:
                # load model parameters if available
                logging.info(f'Loading existing teacher model params from {init_model}')
                teacher_states = self.load(init_model)
            else:
                teacher_states = {}
            
            if init_student_model is not None:
                # load model parameters if available
                logging.info(f'Loading existing student model params from {init_student_model}')
                student_states = self.load(init_student_model)
            else:
                student_states = {}

        if shared is not None:
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        elif self._should_initialize_optimizer():
            # do this regardless of share state, but don't
            if student_states.get('optimizer', False) and student_states.get('optimizer_type', False):
                self.init_optim(
                    [p for p in self.student_model.parameters() if p.requires_grad],
                    optim_states=student_states.get('optimizer'),
                    saved_optim_type=student_states.get('optimizer_type'),
                )
                self.build_lr_scheduler(student_states, hard_reset=is_finetune)
            elif teacher_states.get('optimizer', Flase) and teacher_states.get('optimizer_type', False):
                self.init_optim(
                    [p for p in self.student_model.parameters() if p.requires_grad],
                    optim_states=teacher_states.get('optimizer'),
                    saved_optim_type=teacher_states.get('optimizer_type'),
                )
                self.build_lr_scheduler(teacher_states, hard_reset=is_finetune)

        if shared is None and is_distributed():
            device_ids = None if self.model_parallel else [self.opt['gpu']]
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=device_ids, broadcast_buffers=False
            )
            self.student_model = torch.nn.parallel.DistributedDataParallel(
                self.student_model, device_ids=device_ids, broadcast_buffers=False
            )

        self.reset()

        opt = self.opt

        self.token_criterion = self.build_token_criterion()

    
    def state_dict(self):
        """
        Get the state dict for saving.

        Override this method for more specific saving.
        """
        states = {}
        
        if hasattr(self, 'student_model'):
            if hasattr(self.model, 'module'):
                # did we wrap in a DistributedDataParallel
                states['student_model'] = self.model.module.state_dict()
            else:
                states['student_model'] = self.model.state_dict()

        if hasattr(self, 'optimizer'):
            # save optimizer params
            states['optimizer'] = self.optimizer.state_dict()
            states['optimizer_type'] = self.opt['optimizer']

        # lr scheduler
        states['number_training_updates'] = self._number_training_updates
        if getattr(self, 'scheduler', None):
            states['lr_scheduler'] = self.scheduler.get_state_dict()
            states['lr_scheduler_type'] = self.opt['lr_scheduler']
            states['warmup_scheduler'] = self.scheduler.get_warmup_state_dict()

        return states

    def share(self):
        """
        Share fields from parent as well as useful objects in this class.

        Subclasses will likely want to share their model as well.
        """
        shared = super().share()
        if self.opt.get('numthreads', 1) > 1 and isinstance(self.metrics, dict):
            self.student_model.share_memory()
        shared['student_model'] = self.student_model
        shared['model'] = self.model
        return shared

    def _get_student_init_model(self, opt: Opt, shared):
        init_model = None
        is_fintuning = True
        if not shared:  # only do this on first setup
            # first check load path in case we need to override paths
            if opt.get('init_model_student') and os.path.isfile(opt['init_model_student']):
                # check first for 'init_model' for loading model from file
                init_model = opt['init_model_student']
            if opt.get('student_model_file') and os.path.isfile(opt['student_model_file']):
                # next check for 'model_file', this would override init_model
                init_model = opt['student_model_file']
            if (
                opt.get('load_from_checkpoint')
                and opt.get('init_model_student')
                and opt['init_model_student'].endswith('.checkpoint')
            ):
                # but if we're loading from a checkpoint, we should explicitly load
                # from that point
                init_model = opt['init_model']

            if init_model is not None:
                # if we are loading a model, should load its dict too
                if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                    opt['student_dict_file'] = init_model + '.dict'
                
        return init_model, is_fintuning


    def _get_init_model(self, opt: Opt, shared):
        """
        Get model file to initialize with.

        If `init_model` exits, we will return the path to that file and maybe
        load dict file from that path. Otherwise, use `model_file.`

        :return:  path to load model from, whether we loaded from `init_model`
                  or not
        """
        init_model = None
        if not shared:  # only do this on first setup
            # first check load path in case we need to override paths
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                # check first for 'init_model' for loading model from file
                init_model = opt['init_model']
                is_finetune = True
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # next check for 'model_file', this would override init_model
                init_model = opt['model_file']
                is_finetune = False
            if (
                opt.get('load_from_checkpoint')
                and opt.get('init_model')
                and opt['init_model'].endswith('.checkpoint')
            ):
                # but if we're loading from a checkpoint, we should explicitly load
                # from that point
                init_model = opt['init_model']
                is_finetune = False

            if init_model is not None:
                # if we are loading a model, should load its dict too
                if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                    opt['dict_file'] = init_model + '.dict'

        return init_model

    def save(self, path=None):
        """
        Save model parameters to path (or default to model_file arg).

        Please try to refrain from overriding this function, and instead override
        `state_dict(self)` for more specific saving.
        """
        path = self.opt.get('student_model_file', None) if path is None else path

        if path:
            model_dict_path = path + '.dict'
            if hasattr(self, 'dict') and not os.path.exists(
                model_dict_path
            ):  # force save dictionary
                # TODO: Look into possibly overriding opt('dict_file') with new path
                logging.debug(f'Saving dictionary to {model_dict_path}')
                self.dict.save(model_dict_path, sort=False)
            states = self.state_dict()
            if states:  # anything found to save?
                atomic_save(states, path)
                # save opt file
                self.opt.save(path + '.opt')

    def load_state_dict(self, state_dict, student= True):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        if not student:
            super().load_state_dict(self, state_dict)
        else:
            try:
                self.student_model.load_state_dict(state_dict)
            except RuntimeError as msg:
                msg_ = str(msg)
                if 'size mismatch' in msg_ and 'embedding' in msg_:
                    if hasattr(self, 'special_toks') and len(self.special_toks) > 0:
                        state_dict = self._resize_token_embeddings(state_dict, msg_)
                        self.student_model.load_state_dict(state_dict)
                        self.resized_embeddings = True  # make note that we resized here
                    else:
                        raise RuntimeError(
                            f'{msg_}\n'
                            '-----------------\n'
                            'Could not load the model due to a size mismatch in the '
                            'embeddings. A common reason for this is trying to load '
                            'a model trained with fp16 but loaded without fp16. Try '
                            'adding --fp16 true or --force-fp16-tokens true.'
                        )
                else:
                    raise
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Return opt and model states.

        Override this method for more specific loading.
        """
        import parlai.utils.pickle

        states = torch.load(
            path, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
        )
        if 'student_model' in states:
            self.load_state_dict(states['student_model'])
        if 'model' in states:
            self.load_state_dict(states['model'], False)
        if 'optimizer' in states and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(states['optimizer'])
        return states

    def _create_soft_labels(self, output_vec):
        """
        create soft target labels for student network from teacher network
        output_vec: word vector generated by teacher network
        """
        label_vecs = []
        for label_vec in output_batch:
            label_vec = self._check_truncate(label_vec, self.label_truncate, True)
            label_vecs.append(torch.LongTensor(label_vec))

        some_labels_avail = True if len(labels) else False

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'labels'

            y_lens = [y.shape[0] for y in label_vecs]
            ys, y_lens = self._pad_tensor(label_vecs)
        return ys, y_lens

    def batchify(self, obs_batch, sort=False):
        """
        Create a batch of valid observations from an unchecked batch.

        A valid observation is one that passes the lambda provided to the
        function, which defaults to checking if the preprocessed 'text_vec'
        field is present which would have been set by this agent's 'vectorize'
        function.

        Returns a namedtuple Batch. See original definition above for in-depth
        explanation of each field.

        If you want to include additonal fields in the batch, you can subclass
        this function and return your own "Batch" namedtuple: copy the Batch
        namedtuple at the top of this class, and then add whatever additional
        fields that you want to be able to access. You can then call
        super().batchify(...) to set up the original fields and then set up the
        additional fields in your subclass and return that batch instead.

        :param obs_batch:
            List of vectorized observations

        :param sort:
            Default False, orders the observations by length of vectors. Set to
            true when using torch.nn.utils.rnn.pack_padded_sequence.  Uses the text
            vectors if available, otherwise uses the label vectors if available.
        """
        if len(obs_batch) == 0:
            return Batch(batchsize=0)

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch(batchsize=0)

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs, x_lens = None, None
        if any(ex.get('text_vec') is not None for ex in exs):
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = self._pad_tensor(_xs)
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = labels_avail or any('eval_labels_vec' in ex for ex in exs)

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]

            ys, y_lens = self._pad_tensor(label_vecs)

            if sort and xs is None:
                ys, valid_inds, label_vecs, labels, y_lens = argsort(
                    y_lens, ys, valid_inds, label_vecs, labels, y_lens, descending=True
                )

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return Batch(
            batchsize=len(valid_inds),
            text_vec=xs,
            text_lengths=x_lens,
            label_vec=ys,
            label_lengths=y_lens,
            labels=labels,
            valid_indices=valid_inds,
            candidates=cands,
            candidate_vecs=cand_vecs,
            image=imgs,
            observations=exs,
        )
    
    def act(self):
        """
        Call batch_act with the singleton batch.
        """
        # BatchWorld handles calling self_observe, but we're in a Hogwild or Interactive
        # world, so we need to handle this ourselves.
        response = self.batch_act([self.observation])[0]
        self.self_observe(response)
        return response

    def batch_act(self, observations):
        """
        Process a batch of observations (batchsize list of message dicts).

        These observations have been preprocessed by the observe method.

        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        # clear local metrics before anything else
        self._local_metrics.clear()

        # initialize a list of replies with this agent's id
        batch_reply = [
            Message({'id': self.getID(), 'episode_done': False}) for _ in observations
        ]

        # create a batch from the vectors
        batch = self.batchify(observations)
        
        with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back gradients.
            output , soft_target_vecs = self.teacher_eval_step(batch)

        ys, ys_lens = self._create_soft_labels(soft_target_vecs)

        # setting teacher labels into batch obj
        batch.labels = ys
        batch.label_lengths = ys_lens

        # check if there are any labels available, if so we will train on them
        self.is_training = True if batch.labels else False

        self.global_metrics.add('exps', GlobalTimerMetric(batch.batchsize))

        if (
            'label_vec' in batch
            and 'text_vec' in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            # tokens per batch
            # we divide by the binary is_primary_worker() so that the numerator is
            # num_tokens in all workers, and the denominator is 1.
            lt = (batch.label_vec != self.NULL_IDX).sum().item()
            ltpb = GlobalAverageMetric(lt, float(is_primary_worker()))
            self.global_metrics.add('ltpb', ltpb)
            self.global_metrics.add('ltps', GlobalTimerMetric(lt))

            ct = (batch.text_vec != self.NULL_IDX).sum().item()
            ctpb = GlobalAverageMetric(ct, float(is_primary_worker()))
            self.global_metrics.add('ctpb', ctpb)
            self.global_metrics.add('ctps', GlobalTimerMetric(ct))

            ttpb = GlobalAverageMetric(ct + lt, float(is_primary_worker()))
            self.global_metrics.add('tpb', ttpb)
            self.global_metrics.add('tps', GlobalTimerMetric(ct + lt))

        if self.is_training:
            # register the start of updates for later counting when they occur
            self.global_metrics.add('ups', GlobalTimerMetric(0))
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back gradients.
                output = self.eval_step(batch)

        if output is not None:
            # local metrics are automatically matched up
            self.match_batch(batch_reply, batch.valid_indices, output)

        # broadcast the metrics back
        for k, values in self._local_metrics.items():
            if len(values) != len(batch.valid_indices):
                raise IndexError(
                    f"Batchsize mismatch on metric {k} (got {len(values)}, "
                    f"expected {len(batch.valid_indices)}"
                )
            for i, value in zip(batch.valid_indices, values):
                if 'metrics' not in batch_reply[i]:
                    batch_reply[i]['metrics'] = {}
                batch_reply[i]['metrics'][k] = value

        # register the end of timers
        endtimer = GlobalTimerMetric(0)
        self.global_metrics.add('exps', endtimer)
        if (
            'label_vec' in batch
            and 'text_vec' in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            self.global_metrics.add('ltps', GlobalTimerMetric(0))
            self.global_metrics.add('ctps', GlobalTimerMetric(0))
            self.global_metrics.add('tps', GlobalTimerMetric(0))

        # Make sure we push all the metrics to main thread in hogwild/workers
        self.global_metrics.flush()

        return batch_reply

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        teacher = False
    ):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.

        :return:
            tuple (beam_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        model = self.model if teacher else self.student_model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module if teacher else self.student_model.module

        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = (
            len(batch.text_lengths)
            if batch.text_lengths is not None
            else len(batch.image)  # type: ignore
        )
        if batch.text_vec is not None:
            batchsize = batch.text_vec.size(0)
            beams = [
                self._treesearch_factory(dev)
                .set_context(self._get_context(batch, batch_idx))
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts].unsqueeze(-1).repeat(1, beam_size)
                prefix_score = score.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.NULL_IDX)
                score[prefix_mask] = neginf(score.dtype)
                score[prefix_mask] = score[prefix_mask].scatter_(
                    -1,
                    prefix_toks[prefix_mask].unsqueeze(-1),
                    prefix_score[prefix_mask],
                )
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]
        if self.opt.get('verbose'):
            for i, beams in enumerate(n_best_beam_preds_scores):
                for b, (tokens, score) in enumerate(beams):
                    gen = self._v2t(tokens)
                    logging.debug(f"Batch[{i:3d}] Beam[{b:3d}]: ({score:4.2f}): {gen}")
                logging.debug('-')

        return beam_preds_scores, beams
    
    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.student_model.eval()
        cand_scores = None
        token_losses = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            beam_preds_scores, _ = self._generate(batch, self.beam_size, maxlen, teacher= False)
            preds, scores = zip(*beam_preds_scores)
            self._add_generation_metrics(batch, preds)

        cand_choices = None
        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.student_model.encoder(*self._encoder_input(batch))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.student_model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = self._pad_tensor(batch.candidate_vecs[i])
                scores, _ = self.student_model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = [self._v2t(p) for p in preds] if preds is not None else None
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
            self._compute_nltk_bleu(batch, text)
        return Output(text, cand_choices, token_losses=token_losses)
    
    def teacher_eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            beam_preds_scores, _ = self._generate(batch, self.beam_size, maxlen, teacher= True)
            preds, scores = zip(*beam_preds_scores)
            self._add_generation_metrics(batch, preds)

        cand_choices = None
        # TODO: abstract out the scoring here

        text = [self._v2t(p) for p in preds] if preds is not None else None
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
            self._compute_nltk_bleu(batch, text)
        return Output(text, cand_choices, token_losses=token_losses), preds

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.student_model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss
        
    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        # helps with memory usage
        # note we want to use the opt's batchsize instead of the observed batch size
        # in case dynamic batching is in use
        self._init_cuda_buffer(self.opt['batchsize'], self.label_truncate or 256)
        # setting student model to train mode and teacher model in eval mode
        self.student_model.train()
        self.model.eval()
        self.zero_grad()

        try:
            loss = self.compute_loss(batch)
            self.backward(loss)
            self.update_params()
            oom_sync = False
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                oom_sync = True
                logging.error(
                    'Ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                self.global_metrics.add('skipped_batches', SumMetric(1))
            else:
                raise e

        if oom_sync:
            # moved outside of the try-except because the raised exception in scope
            # actually prevents from the data being freed, which can sometimes cause
            # us to OOM during our OOM handling.
            # https://github.com/pytorch/pytorch/issues/18853#issuecomment-583779161

            # gradients are synced on backward, now this model is going to be
            # out of sync! catch up with the other workers
            self._init_cuda_buffer(8, 8, True)
    

class DistillGeneratorModel(TorchGeneratorModel, ABC):
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

    def decode_forced(self, encoder_states, ys):
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
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds, latent

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
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
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)

        # use teacher forcing
        scores, preds, latent= self.decode_forced(encoder_states, ys)
        return scores, preds, encoder_states, latent