import time
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import BlenderbotTokenizer
from modules import TransformerGeneratorModel
from inference import Inference
from opt import params, inference_params


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mname = 'facebook/blenderbot-3B'
tokenizer = BlenderbotTokenizer.from_pretrained(mname)
model = TransformerGeneratorModel(params, tokenizer)

checkpoint = torch.load("model_pruned")
model.load_state_dict(checkpoint['model'])
model = model.half()
model = model.to(device).eval()

print(f"Data type of model parameters: {next(model.parameters()).dtype}")

with open('data/train.txt') as f:
    turns = [str_to_turn(l.strip()) for l in tqdm(f.readlines())]

feed_text = [t['text'] for t in turns]

conversation_times = []
for _ in tqdm(range(100)):
    chat = Inference(model, tokenizer, device=device, **inference_params)
    turn_times = []
    for _ in range(10):
        text = random.choice(feed_text)
        start_time = time.time()
        text = chat.converse(text)
        turn_times.append(time.time() - start_time)
    conversation_times.append(turn_times)

print(f"Average inference time: {np.array(conversation_times).mean()}")
