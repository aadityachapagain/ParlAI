import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BlenderbotTokenizer
from modules import TransformerGeneratorModel
from opt import params


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
                 text_truncate=128,
                 label_truncate=None):
        if label_truncate is None:
            label_truncate = text_truncate

        with open(parlai_format_data_path) as f:
            turns = [str_to_turn(l.strip()) for l in f.readlines()]

        self.data = []

        history = []
        label_tokens = []
        for turn in tqdm(turns):
            history += label_tokens
            for text in turn['text'].split('\n'):
                text_tokens = tokenizer(text,
                                        padding=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False)['input_ids']
                history += text_tokens
                history += [tokenizer.eos_token_id, ]

            label_tokens = tokenizer(turn['labels'][0],
                                     padding=False,
                                     return_token_type_ids=False,
                                     return_attention_mask=False)['input_ids']
            label_tokens += [tokenizer.eos_token_id, ]

            self.data.append((torch.tensor(history[-text_truncate:]), torch.tensor(label_tokens[-label_truncate:])))

            if turn['episode_done']:
                history = []
                label_tokens = []

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


mname = 'facebook/blenderbot-3B'
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

dataset = ParlaiFormatDataset('data/train.txt',
                              tokenizer,
                              text_truncate=128)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,
                                         collate_fn=Collator(tokenizer))

model = TransformerGeneratorModel(params, tokenizer)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


print(f"Number of parameters: {count_parameters(model)}")
print(f"Transferring model to {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id,
                                reduction='none')
optimizer = torch.optim.Adamax(model.parameters(),
                               lr=7.5e-06)


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

    return loss, token_acc


# training
model.train()

log_every_step = 200

for epoch in range(5):
    running_loss = 0.0
    running_token_acc = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        scores, preds, *_ = model(inputs, ys=target)

        batch_loss, batch_token_acc = compute_loss_and_metrics(scores, preds, target, tokenizer.pad_token_id)
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item()
        running_token_acc += batch_token_acc.item()
        if (i % log_every_step) == (log_every_step - 1):
            print('[%d, %5d] loss: %.3f; token acc: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_every_step, running_token_acc / log_every_step))
            running_loss = 0.0
            running_token_acc = 0.0

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                'model.pt',
            )
