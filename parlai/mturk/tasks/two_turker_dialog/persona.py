import os
import random
import numpy as np
import pandas as pd

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'child_persona_permutation.csv'))


class Persona:
    def __init__(self, child_personas=None, robot_personas=None, talk_theme=None):
        if child_personas is None:
            child_personas = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'child_persona_permutation.csv'))
        self.child_personas = child_personas
        if robot_personas is None:
            robot_personas = [
                {
                    "title": "Shy and Unsure",
                    "description": "Karu is shy, cautious and feels a bit unsure about being able to accomplish its mission."
                },
                {
                    "title": "Curious and Dedicated",
                    "description": "Karu is curious about learning the human world and dedicated to its core mission."
                },
                {
                    "title": "Optimistic and Vulnerable",
                    "description": "Karu is optimistic about the future and completely aware of its personal limitations and challenges."
                },
                {
                    "title": "Thoughtful, Supportive and Goal Oriented",
                    "description": "Karu values learning and the ability to make quantifiable progress every day - small steps can take you to amazing places."
                },
                {
                    "title": "Grateful and Reflective",
                    "description": "Karu likes to pause, reflect, and acknowledge moments of growth and express appreciation for things it finds of interest."
                },
            ]
        self.robot_personas = robot_personas
        if talk_theme is None:
            talk_theme = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'talk_theme.csv'))
        self.talk_theme = talk_theme
        self.talk_theme_probabilities = [0.7, 0.3]

    def gen_child_persona(self):
        gen = random.choice(self.child_personas['Gender'].dropna())
        age = random.choice(self.child_personas['Age'].dropna())
        name = random.choice(self.child_personas['Male Names'].dropna()) if gen == 'boy' else random.choice(
            self.child_personas['Female Names'].dropna())
        emotion = random.choice(self.child_personas['Emotion'].dropna())
        interest = random.choice(self.child_personas['Interest'].dropna())
        verb = random.choice(self.child_personas['Joins'].dropna())
        return ' '.join([name, 'is', emotion, str(age), 'year old', gen, verb, interest])

    def gen_robot_persona(self):
        robot = random.choice(self.robot_personas)
        robot_persona_text = (
            f'{robot["title"]} Karu. '
            f'{robot["description"]}'
        )
        return robot_persona_text

    def gen_talk_theme(self):
        theme_type = np.random.choice(self.talk_theme.columns, p=self.talk_theme_probabilities)
        theme = random.choice(self.talk_theme[theme_type].dropna())
        if theme_type == 'Talk Theme':
            theme_sentence = f'You want to talk about {theme}.'
        else:
            theme_sentence = f'You are worried about {theme}.'
        return theme, theme_type, theme_sentence
