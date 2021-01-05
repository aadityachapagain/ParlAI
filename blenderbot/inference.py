import torch
import torch.nn.functional as F
from modules import TransformerGeneratorModel
from search_algorithms import GreedySearch, BeamSearch, DelayedBeamSearch


class Inference:
    def __init__(self,
                 model: TransformerGeneratorModel,
                 tokenizer,
                 method,
                 beam_size,
                 min_beam_length,
                 beam_block_ngram,
                 beam_context_block_ngram,
                 beam_length_penalty,
                 input_maxlen,
                 max_len,
                 device,
                 topk=10,
                 beam_delay=30
                 ):
        self.model = model
        self.beam_size = beam_size
        self.method_name = method
        self.min_beam_length = min_beam_length
        self.beam_block_ngram = beam_block_ngram
        self.beam_context_block_ngram = beam_context_block_ngram
        self.beam_length_penalty = beam_length_penalty
        self.max_len = max_len
        self.device = device
        self.topk = topk
        self.beam_delay = beam_delay
        self.tokenizer = tokenizer
        self.input_maxlen = input_maxlen
        self.history = torch.tensor([], dtype=torch.int64).to(self.device)
        self.eos_token_id = torch.tensor([self.tokenizer.eos_token_id, ], dtype=torch.int64).to(self.device)
        self.delimiter = torch.tensor([228, 228], dtype=torch.int64).to(self.device)

    def get_search_method(self):
        if self.method_name == 'greedy':
            return GreedySearch(
                self.beam_size,
                min_length=self.min_beam_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.beam_length_penalty,
                padding_token=self.model.pad_idx,
                bos_token=self.model.start_idx,
                eos_token=self.model.end_idx,
                device=self.device
            )
        elif self.method_name == 'beam':
            return BeamSearch(
                self.beam_size,
                min_length=self.min_beam_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.beam_length_penalty,
                padding_token=self.model.pad_idx,
                bos_token=self.model.start_idx,
                eos_token=self.model.end_idx,
                device=self.device
            )
        elif self.method_name == 'delayedbeam':
            return DelayedBeamSearch(
                self.topk,
                self.beam_delay,
                self.beam_size,
                min_length=self.min_beam_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.beam_length_penalty,
                padding_token=self.model.pad_idx,
                bos_token=self.model.start_idx,
                eos_token=self.model.end_idx,
                device=self.device
            )
        else:
            raise ValueError(f"Can't use inference method {self.method_name}")

    def _get_initial_decoder_input(self, batch_size):
        return (
            torch.LongTensor([self.model.start_idx])
                .expand(batch_size * self.beam_size, 1)
                .to(self.device)
        )

    def generate(self, batch_input):
        batch_input = batch_input.to(self.device)
        with torch.no_grad():
            encoder_states = self.model.encoder(batch_input)
            batch_size = batch_input.size(0)
            beams = [self.get_search_method().set_context(batch_input[batch_idx])
                     for batch_idx in range(batch_size)]
            decoder_input = self._get_initial_decoder_input(batch_size)
            inds = torch.arange(batch_size).to(self.device).unsqueeze(1).repeat(1, self.beam_size).view(-1)
            encoder_states = self.model.reorder_encoder_states(encoder_states, inds)
            incr_state = None
            for _ts in range(self.max_len):
                if all([b.is_done() for b in beams]):
                    break
                score, incr_state = self.model.decoder(decoder_input, encoder_states, incr_state)
                score = score[:, -1:, :]
                score = self.model.output(score)
                score = score.view(batch_size, self.beam_size, -1)
                score = F.log_softmax(score, dim=-1, dtype=torch.float32)

                for i, b in enumerate(beams):
                    if not b.is_done():
                        b.advance(score[i])

                incr_state_inds = torch.cat(
                    [
                        self.beam_size * i + b.get_backtrack_from_current_step()
                        for i, b in enumerate(beams)
                    ]
                )
                incr_state = self.model.reorder_decoder_incremental_state(
                    incr_state, incr_state_inds
                )
                selection = torch.cat(
                    [b.get_output_from_current_step() for b in beams]
                ).unsqueeze(-1)
                decoder_input = self._get_next_decoder_input(
                    decoder_input, selection, incr_state_inds
                )

            n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]
            beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]
            preds, scores = zip(*beam_preds_scores)
            return preds, scores

    def _get_next_decoder_input(self, prev_input, selection, incr_state_inds):
        prev_input = torch.index_select(prev_input, 0, incr_state_inds)
        decoder_input = torch.cat([prev_input, selection], dim=-1)
        return decoder_input

    def converse(self, text_input):
        tokens = self.tokenizer(text_input,
                                padding=False,
                                return_token_type_ids=False,
                                return_attention_mask=False,
                                add_special_tokens=False)['input_ids']
        tokens = torch.tensor(tokens).to(self.device)
        self.history = torch.cat((self.history, tokens, self.eos_token_id), dim=0)
        self.history = self.history[-(self.input_maxlen-1):]
        preds, scores = self.generate(self.history.view(1, -1))
        preds = preds[0]
        text = self.tokenizer.decode(preds, skip_special_tokens=True)

        self.history = torch.cat((self.history[:-1], self.delimiter, preds[1:-1], self.delimiter), dim=0)
        return text
