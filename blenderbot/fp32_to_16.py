import torch
from transformers import BlenderbotTokenizer
from modules import TransformerGeneratorModel
from opt import params

mname = 'facebook/blenderbot-3B'
tokenizer = BlenderbotTokenizer.from_pretrained(mname)
model = TransformerGeneratorModel(params, tokenizer)

checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model = model.half()

torch.save(
    {
        'model_state_dict': model.state_dict(),
    },
    'model_half.pt',
)
