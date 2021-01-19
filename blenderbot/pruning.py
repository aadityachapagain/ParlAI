from tqdm import tqdm
from heapq import heappush, heappop
import torch
import torch.nn as nn
from modules import TransformerGeneratorModel
from opt import params
from transformers import BlenderbotSmallTokenizer, BlenderbotTokenizer


def str_to_turn(txt, ignore_fields=''):
    """convert parlai format line to python dictionary"""

    def tostr(txt):
        txt = str(txt)
        txt = txt.replace('\\t', '\t')
        txt = txt.replace('\\n', '\n')
        txt = txt.replace('__PIPE__', '|')
        return txt

    def tolist(txt):
        vals = txt.split('|')
        for v in vals:
            v = tostr(v)
        return vals

    def convert(key, value):
        if key == 'text' or key == 'id':
            return tostr(value)
        elif (
                key == 'label_candidates'
                or key == 'labels'
                or key == 'eval_labels'
                or key == 'text_candidates'
        ):
            return tolist(value)
        elif key == 'episode_done':
            return bool(value)
        else:
            return tostr(value)

    if txt == '' or txt is None:
        return None

    turn = {}
    for t in txt.split('\t'):
        ind = t.find(':')
        key = t[:ind]
        value = t[ind + 1:]
        if key not in ignore_fields.split(','):
            turn[key] = convert(key, value)
    turn['episode_done'] = turn.get('episode_done', False)
    return turn


class ParlaiFormatDataset(torch.utils.data.Dataset):
    def __init__(self,
                 parlai_format_data_path,
                 tokenizer,
                 delimiter=[2, ],
                 text_truncate=128,
                 label_truncate=None):
        if label_truncate is None:
            label_truncate = text_truncate

        with open(parlai_format_data_path) as f:
            turns = [str_to_turn(l.strip()) for l in f.readlines()]

        self.data = []

        history = []
        idx = 0
        for turn in tqdm(turns):
            text_tokens = tokenizer(turn['text'],
                                    padding=False,
                                    return_token_type_ids=False,
                                    return_attention_mask=False,
                                    add_special_tokens=False)['input_ids']
            history += text_tokens + [tokenizer.eos_token_id, ]

            label_tokens = tokenizer(turn['labels'][0],
                                     padding=False,
                                     return_token_type_ids=False,
                                     return_attention_mask=False,
                                     add_special_tokens=False)['input_ids']
            label_tokens += [tokenizer.eos_token_id, ]

            self.data.append((torch.tensor(history[-text_truncate:]), torch.tensor(label_tokens[-label_truncate:])))

            history = history[:-1] + delimiter + label_tokens[:-1]

            if turn['episode_done']:
                history = []
            idx += 1
            if idx == 1000:
                break

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        x, y = zip(*batch)
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        y = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return x, y


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mname = 'facebook/blenderbot-90M'
tokenizer = BlenderbotSmallTokenizer.from_pretrained(
    mname) if mname == 'facebook/blenderbot-90M' else BlenderbotTokenizer.from_pretrained(mname)
dataset = ParlaiFormatDataset('data/train.txt',
                              tokenizer,
                              text_truncate=512)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                         collate_fn=Collator(tokenizer))
model = TransformerGeneratorModel(params, tokenizer).to(device)
model.load_state_dict(torch.load("/home/sagar/Desktop/parlaiMturk/ParlAI/data/models/blender/blender_90M/model")['model'])
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id,
                                reduction='none')


def compute_loss_and_metrics(scores, preds, target, null_idx):
    loss = criterion(scores.view(-1, scores.size(-1)),
                     target.view(-1))
    loss = loss.view(scores.shape[:-1]).sum(dim=1)

    notnull_tokens = target.ne(null_idx)
    target_tokens = notnull_tokens.long().sum(dim=-1)
    correct_tokens = ((target == preds) * notnull_tokens).sum(dim=-1)

    loss = loss.sum()
    loss /= target_tokens.sum()

    token_acc = correct_tokens.sum() / target_tokens.sum()

    return loss, token_acc, target_tokens


# get the model ffn weights and biases
encoder_inter_weights = torch.zeros(params['n_encoder_layers'],
                                    params['ffn_size'],
                                    params['embedding_size']).to(device)
encoder_inter_biases = torch.zeros(params['n_encoder_layers'],
                                   params['ffn_size'],
                                   ).to(device)
encoder_output_weights = torch.zeros(params['n_encoder_layers'],
                                     params['embedding_size'],
                                     params['ffn_size']).to(device)
encoder_output_biases = torch.zeros(params['n_encoder_layers'],
                                    params['embedding_size']).to(device)

decoder_inter_weights = torch.zeros(params['n_decoder_layers'],
                                    params['ffn_size'],
                                    params['embedding_size']).to(device)
decoder_inter_biases = torch.zeros(params['n_decoder_layers'],
                                   params['ffn_size'],
                                   ).to(device)
decoder_output_weights = torch.zeros(params['n_decoder_layers'],
                                     params['embedding_size'],
                                     params['ffn_size']).to(device)
decoder_output_biases = torch.zeros(params['n_decoder_layers'],
                                    params['embedding_size']).to(device)

for layer_num in range(params['n_encoder_layers']):
    encoder_inter_weights[layer_num] = model.encoder.layers[layer_num].ffn.lin1.weight.detach().to(device)
    encoder_inter_biases[layer_num] = model.encoder.layers[layer_num].ffn.lin1.bias.detach().to(device)
    encoder_output_weights[layer_num] = model.encoder.layers[layer_num].ffn.lin2.weight.detach().to(device)
    encoder_output_biases[layer_num] = model.encoder.layers[layer_num].ffn.lin2.bias.detach().to(device)

for layer_num in range(params['n_decoder_layers']):
    decoder_inter_weights[layer_num] = model.decoder.layers[layer_num].ffn.lin1.weight.detach().to(device)
    decoder_inter_biases[layer_num] = model.decoder.layers[layer_num].ffn.lin1.bias.detach().to(device)
    decoder_output_weights[layer_num] = model.decoder.layers[layer_num].ffn.lin2.weight.detach().to(device)
    decoder_output_biases[layer_num] = model.decoder.layers[layer_num].ffn.lin2.bias.detach().to(device)

encoder_head_importance = torch.zeros(params['n_encoder_layers'], params['n_heads']).to(device)
decoder_head_importance = torch.zeros(params['n_decoder_layers'], params['n_heads']).to(device)

encoder_ffn_importance = torch.zeros(params['n_encoder_layers'], params['ffn_size']).to(device)
decoder_ffn_importance = torch.zeros(params['n_decoder_layers'], params['ffn_size']).to(device)

encoder_head_mask = torch.ones(params['n_encoder_layers'], params['n_heads'], requires_grad=True).to(device)
decoder_head_mask = torch.ones(params['n_decoder_layers'], params['n_heads'], requires_grad=True).to(device)

model.eval()
for inputs, target in tqdm(dataloader):
    inputs = inputs.to(device)
    target = inputs.to(device)

    scores, preds, *_ = model(inputs, ys=target,
                              encoder_head_mask=encoder_head_mask,
                              decoder_head_mask=decoder_head_mask)
    batch_loss, *_, = compute_loss_and_metrics(scores, preds, target, tokenizer.pad_token_id)
    batch_loss.backward()

    encoder_head_importance += encoder_head_mask.grad.abs().detach()
    decoder_head_importance += decoder_head_mask.grad.abs().detach()

    for layer_num in range(params['n_encoder_layers']):
        encoder_ffn_importance[layer_num] += torch.abs(
            torch.sum(model.encoder.layers[layer_num].ffn.lin1.weight.grad.detach() * encoder_inter_weights[layer_num],
                      1)
            + model.encoder.layers[layer_num].ffn.lin1.bias.grad.detach() * encoder_inter_biases[layer_num]
        )

    for layer_num in range(params['n_decoder_layers']):
        decoder_ffn_importance[layer_num] += torch.abs(
            torch.sum(model.decoder.layers[layer_num].ffn.lin1.weight.grad.detach() * decoder_inter_weights[layer_num],
                      1)
            + model.decoder.layers[layer_num].ffn.lin1.bias.grad.detach() * decoder_inter_biases[layer_num]
        )

# TODO divide head importance by tot_tokens

# Layerwise importance normalization
encoder_head_importance /= torch.pow(torch.pow(encoder_head_importance, 2).sum(-1), 1 / 2).unsqueeze(-1) + 1e-20
decoder_head_importance /= torch.pow(torch.pow(decoder_head_importance, 2).sum(-1), 1 / 2).unsqueeze(-1) + 1e-20

# rewire network
target_num_heads = 8
target_ffn_size = 1024

encoder_head_importance = encoder_head_importance.cpu()
decoder_head_importance = decoder_head_importance.cpu()
encoder_ffn_importance = encoder_ffn_importance.cpu()
decoder_ffn_importance = decoder_ffn_importance.cpu()


def sort_by_importance(weight, bias, importance, num_instances, stride):
    importance_ordered = []
    i = 0
    for heads in importance:
        heappush(importance_ordered, (-heads, i))
        i += 1
    sorted_weight_to_concat = None
    sorted_bias_to_concat = None
    i = 0
    while importance_ordered and i < num_instances:
        head_to_add = heappop(importance_ordered)[1]
        if sorted_weight_to_concat is None:
            sorted_weight_to_concat = (weight.narrow(0, int(head_to_add * stride), int(stride)),)
        else:
            sorted_weight_to_concat += (weight.narrow(0, int(head_to_add * stride), int(stride)),)
        if bias is not None:
            if sorted_bias_to_concat is None:
                sorted_bias_to_concat = (bias.narrow(0, int(head_to_add * stride), int(stride)),)
            else:
                sorted_bias_to_concat += (bias.narrow(0, int(head_to_add * stride), int(stride)),)
        i += 1
    return torch.cat(sorted_weight_to_concat), torch.cat(
        sorted_bias_to_concat) if sorted_bias_to_concat is not None else None


def rewire_network(module,
                   num_layers,
                   embedding_size,
                   n_heads,
                   head_importance,
                   ffn_importance,
                   target_num_heads,
                   target_ffn_size):
    for layer_num in range(num_layers):
        query_weight = module.layers[layer_num].attention.q_lin.weight
        query_bias = module.layers[layer_num].attention.q_lin.bias
        key_weight = module.layers[layer_num].attention.k_lin.weight
        key_bias = module.layers[layer_num].attention.k_lin.bias
        value_weight = module.layers[layer_num].attention.v_lin.weight
        value_bias = module.layers[layer_num].attention.v_lin.bias

        # sort query, key, value based on the confidence scores
        query_weight, query_bias = sort_by_importance(query_weight,
                                                      query_bias,
                                                      head_importance[layer_num],
                                                      target_num_heads,
                                                      embedding_size / n_heads)
        module.layers[layer_num].attention.q_lin.weight = torch.nn.Parameter(query_weight)
        module.layers[layer_num].attention.q_lin.bias = torch.nn.Parameter(query_bias)

        key_weight, key_bias = sort_by_importance(key_weight,
                                                  key_bias,
                                                  head_importance[layer_num],
                                                  target_num_heads,
                                                  embedding_size / n_heads)
        module.layers[layer_num].attention.k_lin.weight = torch.nn.Parameter(key_weight)
        module.layers[layer_num].attention.k_lin.bias = torch.nn.Parameter(key_bias)

        value_weight, value_bias = sort_by_importance(value_weight,
                                                      value_bias,
                                                      head_importance[layer_num],
                                                      target_num_heads,
                                                      embedding_size / n_heads)
        module.layers[layer_num].attention.v_lin.weight = torch.nn.Parameter(value_weight)
        module.layers[layer_num].attention.v_lin.bias = torch.nn.Parameter(value_bias)

        attention_out_weight, attention_out_bias = sort_by_importance(
            module.layers[layer_num].attention.out_lin.weight.transpose(0, 1),
            module.layers[layer_num].attention.out_lin.bias,
            head_importance[layer_num],
            target_num_heads,
            embedding_size / n_heads)
        module.layers[layer_num].attention.out_lin.weight = torch.nn.Parameter(attention_out_weight.transpose(0, 1))
        module.layers[layer_num].attention.out_lin.bias = torch.nn.Parameter(attention_out_bias)

        weight_sorted, bias_sorted = sort_by_importance(
            module.layers[layer_num].ffn.lin1.weight,
            module.layers[layer_num].ffn.lin1.bias,
            ffn_importance[layer_num],
            target_ffn_size,
            1
        )
        module.layers[layer_num].ffn.lin1.weight = torch.nn.Parameter(weight_sorted)
        module.layers[layer_num].ffn.lin1.bias = torch.nn.Parameter(bias_sorted)

        weight_sorted, bias_sorted = sort_by_importance(
            module.layers[layer_num].ffn.lin2.weight.transpose(0, 1),
            module.layers[layer_num].ffn.lin2.bias,
            ffn_importance[layer_num],
            target_ffn_size,
            1
        )
        module.layers[layer_num].ffn.lin1.weight = torch.nn.Parameter(weight_sorted.transpose(0, 1))
        module.layers[layer_num].ffn.lin1.bias = torch.nn.Parameter(bias_sorted)


rewire_network(model.encoder, params['n_encoder_layers'], params['embedding_size'], params['n_heads'],
               encoder_head_importance, encoder_ffn_importance, target_num_heads, target_ffn_size)
rewire_network(model.decoder, params['n_decoder_layers'], params['embedding_size'], params['n_heads'],
               decoder_head_importance, decoder_ffn_importance, target_num_heads, target_ffn_size)

torch.save({'model': model.state_dict()}, 'pruned_model')
