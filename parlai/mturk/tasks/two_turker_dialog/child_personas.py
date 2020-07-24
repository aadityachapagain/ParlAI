import os
import random
import pandas as pd

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'child_persona_permutation.csv'))


def gen_child_persona_sentence():
    gen = random.choice(df['Gender'].dropna())
    age = random.choice(df['Age'].dropna())
    name = random.choice(df['Male Names'].dropna()) if gen == 'boy' else random.choice(df['Female Names'].dropna())
    emotion = random.choice(df['Emotion'].dropna())
    interest = random.choice(df['Interest'].dropna())
    verb = random.choice(df['Joins'].dropna())
    return ' '.join([name, 'is', emotion, str(age), 'year old', gen, verb, interest])


