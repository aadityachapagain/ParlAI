import os
import random
import numpy as np
import pandas as pd


class Context:
    def __init__(self, child_personas=None, robot_personas=None, conv_theme=None):
        if child_personas is None:
            child_personas = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'context_data/child_persona_permutation.csv'))
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

        if conv_theme is None:
            conv_theme = pd.read_csv(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'context_data/talk_theme_and_qualification.csv'))
        self.talk_themes = conv_theme[['Talk about', 'Talk1', 'Talk2', 'Talk3']].dropna().values.tolist()
        self.worry_themes = conv_theme[
            ['Worried about', 'Worry1', 'Worry2', 'Worry3']].dropna().values.tolist()

    def gen_persona_context(self):
        gen = random.choice(self.child_personas['Gender'].dropna())
        age = random.choice(self.child_personas['Age'].dropna())
        name = random.choice(self.child_personas['Male Names'].dropna()) if gen == 'boy' else random.choice(
            self.child_personas['Female Names'].dropna())
        emotion = random.choice(self.child_personas['Emotion'].dropna())
        interest = random.choice(self.child_personas['Interest'].dropna())
        verb = random.choice(self.child_personas['Joins'].dropna())
        child_persona = ' '.join([name, 'is', emotion, str(age), 'year old', gen, verb, interest])

        robot = random.choice(self.robot_personas)
        robot_persona_text = (
            f'{robot["title"]} Karu. '
            f'{robot["description"]}'
        )

        return {
            'child_persona': child_persona,
            'robot_persona': robot_persona_text
        }

    def gen_conv_theme_context(self):
        theme_type = np.random.choice([self.talk_themes, self.worry_themes], p=[0.7, 0.3])
        rand_idx = np.random.choice(range(len(theme_type)))
        theme = theme_type[rand_idx]
        return {
            'theme_type': 'Talk Theme' if theme_type == self.talk_themes else 'Worry Theme',
            'theme': theme[0],
            'theme_sentence': f'you want to talk about {theme[0]}.' if theme_type == self.talk_themes
            else f'You are worried about {theme[0]}.',
            'qual_test_choices': {
                'correct_option': theme[1],
                'option_2': theme[2],
                'option_3': theme[3]
            }
        }

    def gen_context(self):
        return {
            'conv_theme': self.gen_conv_theme_context(),
            'personas': self.gen_persona_context()
        }
