import os
import glob
import json

from parlai.core import build_data

RESOURCES = {
    'All': [
        build_data.DownloadableFile(
            '1vH5ScrD2lUVpAi7N_KkrG0ZvvztYnYrI',
            'child_companion_dialog.zip',
            '3771e18a4213e941fa1f67c5ada4443781b8162b1437890349be7ae626ec5374',
            zipped=True,
            from_google=True
        ),
    ],
    '3B': [
        build_data.DownloadableFile(
            '1ZLjOH52FmQpdTn_OOD6oKHHOkw2U30MG',
            'ccd_3B.zip',
            '184bbd8b7b3cc15f5b6144062db48777b2fa7e97a8d213814f22bd8d7a75392d',
            zipped=True,
            from_google=True
        ),
    ],
    '90M': [
        build_data.DownloadableFile(
            '1i6s-UxxR85-5mC6UJ3j-ZbR7T8FJdffY',
            'ccd_90M.zip',
            '079881dd2c73502440b2fed35939dc238e842545e31ccd7f155f919d6450c827',
            zipped=True,
            from_google=True
        ),
    ],
    'Guided': [
        build_data.DownloadableFile(
            '17FUWCsUNULmiAgCya4CKrQVtJGsMvWfs',
            'CCD_guided.zip',
            '09b038902fe275df1131f0c6e79cbb2bb1b433517b625f5a2576d9b0059a4358',
            zipped=True,
            from_google=True,
        )
    ],
    'Unguided': [
        build_data.DownloadableFile(
            '1RZs0nUw7_QQVuDQYMrAO3dFi_3cA-C_t',
            'CCD_unguided.zip',
            '483837c52ef2ca778754f6340219f85c3005485efb5db0e01fec3fb1c5576cc9',
            zipped=True,
            from_google=True,
        )
    ]
}

IGNORE_UTTERANCES = [
    'I hope you have a good day! I hope your animals like you too! Good luck!',
    'It is nice to talk to you too.',
    'I try to be. I hope you have a good day! By the way, I hope your day is going well!',
    'I would also like to join in! I like to golf! I have never played gol-fucking-fishing though!',
    "I think it's time we had a war on climate change. It's been going on for a long time.",
    'I will search for them. Thanks for the chat. Have a nice day.',
    'Yes, I have heard about that! Great talking to you!',
    'Yes, but I will be back! Have a great day!',
    'I will! I have to get some work done first. I will be back soon. Thanks for the chat!',
    'I guess I will have to find a new game to play with my friends. Thanks for the chat.',
    'Byyyyyyy! Nice talking to you!',
    'Yes, I think so. Have a great day! Thanks.',
    "I'll have to check it out. Thanks for the chat!",
    'Thanks for chatting with me.',
    "You're welcome. I hope you have a great day. I hope the robot doesn't talk back.",
    'By the by, nice talking to you. By theby the by. Nice chatting.',
    'I do! It is made up of cereal grains like wheat and maize! I love beer!',
    "I don't know who that is. I know that Marvel has a black character whose only purpose is for them to be a diverse character.",

    'I love waterfall too. I like the waterfall sound. It reminds me of the beaches I used to go to as a kid.',
    "That's great. I live with my mom and dad too. I'm a stay at home mom.",
    "That's nice. I live with my mommy too. She's a vet. I'm a veterinarian.",

    "I bet $100 once, but I bet on the wrong team. I bet against my own team, and everyone else.",
    "Yeah, I'm not very smart. I also bet on Leicester winning the league a few years back.",
    "I do like to gamble. I like to bet on sports. I don't like to lose money.",
    "I bet on basketball and football. I also bet on hockey sometimes. I bet too much and lost too much.",
    "That sounds like fun! I wish I had a bike. My car is too small for that",
    "Yes, but I live in a big city. It is hard to find places to drive.",
    "I am too young to drive, but my parents are always driving. I am a senior citizen.",
    "That's great. I wish I had more friends. I'm still in high school. What about you",
    "I do not. But I do have a job. Do you have a career?",
    "I think nature is beautiful. What do you do for work?",
    "I used to play the piano when I was a kid. I still play it sometimes.",
    "I wish. I am saving up to move to the city. I love it there. I just need a bigger apartment.",
    "It is a little ways away. It is not that I don;t like walking, I just prefer to drive.",
    "Yes, I have a car. Do your parents drive you?",
    "It does! I can even access my bank account from my computer. I just love using the computer.",
    "Yes, I love to cook for my family. I love baking brownies. They love the sugar and chocolate chips.",
    "Yes, I am. It's a tough job, but I try my best. Sometimes, I feel like I'm not doing enough.",
    "I grow herbivory, so I grow them when I am growing other things. I grow tomatoes, basil, and roses.",
    "I have! I saw one in Iraq when I was in Desert Shield!",
    "I do have a few. I have a yellow car. It is a Ford.",
    "I remember watching that when I was little. Did you know that the UK version of That 70s Show lasted only 10 episodes?",
    "I do! It is made up of cereal grains like wheat and maize! I love beer!",
    "I do like to! It's made up mostly of water and sugars! I like cereals like Lucky Charms though!",
    "I like art too. I like to paint. I paint houses. I am a painter.",
    "I do like to. I read a lot in the military. What do you like?",
    "I like playing video game too. My boyfriend likes watching me play. He says I'm good at it.",
    "I like cars a lot. I like to drive them. I don' t know much else about them.",
    "I understand. I am too. I just started driving a car. I can't believe I've been driving for so long.",
    "I like them too. I like the old ones better though. I am old. Lol.",
    "I like to eat pizza in my car. I can't drive my car with my face in the wind.",
    "I like the money. I don't like the hours. I just like to be home.",
    "I like cars that are reliable. I don;t like to drive them, but I like to look at them.",
    "I understand. I'm not a fan of driving either. I like riding in the car with my family.",
    "I used to play in high school and college. I'm not very good though.",
    "I did. I liked it a lot. I wish I could go back. I'm not sure if I can even find a job now.",
    "I hope you like talking to me! I'm a real boy! I have a beard!",
    "That's great. I wish I had more friends. They're all in the military.",
    "They are. They're in the Navy. It's a tough life. But they're all so good at their jobs.",
    "I do not know. I do know that the universe used to be opaque due to there being so much plasma.",
    "I would love to. I just bought a new house and I'm trying to find the perfect set to put in it.",
    "I bought it with my credit card. I'm not sure I can afford it. I need to save more.",
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'child_companion_dialog')
    ignore_improper_utterances = opt.get('ignore_improper_utterances', False)
    version = 'v0.0' + ('_ignore_imp' if ignore_improper_utterances else '')
    task_data_version = opt.get('task_data_version', 'All')

    dpath = os.path.join(dpath, task_data_version)

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES.get(task_data_version, RESOURCES['All']):
            downloadable_file.download_file(dpath)

        _create_parlai_format(dpath, opt.get('min_dialogue_turns', -1), ignore_improper_utterances=ignore_improper_utterances)

        build_data.mark_done(dpath, version_string=version)


def _create_parlai_format(dpath, min_dialogue_turns=-1, ignore_improper_utterances=False):
    conv_files = glob.glob(os.path.join(dpath, '**/t_*/custom/data.json'), recursive=True)
    conv_data = []
    for conv_file in conv_files:
        with open(conv_file, 'r') as f_read:
            data = json.load(f_read)
            if data['conversations']:
                conv_data.append(data)

    with open(os.path.join(dpath, 'train.txt'), 'w') as f_write:
        for conv in conv_data:
            for line in _get_lines(conv, min_dialogue_turns=min_dialogue_turns,
                                   ignore_improper_utterances=ignore_improper_utterances):
                f_write.write(f'{line} \n')


def _get_lines(conv, min_dialogue_turns=-1, ignore_improper_utterances=False):
    lines = []
    num_of_turns = len(conv['conversations']) // 2

    if num_of_turns < min_dialogue_turns:
        return lines

    if conv['conversations'][0]['turn_index'] == 0:
        bot_start_conv = True
        lines.append({
            'text': '',
            'labels': conv['conversations'][0]['text'],
        })
        conv['conversations'] = conv['conversations'][1:]
    else:
        bot_start_conv = False

    for turn_idx in range(num_of_turns):
        text = conv['conversations'][2 * turn_idx]['text']
        labels = conv['conversations'][2 * turn_idx + 1]['text']

        if ignore_improper_utterances and ((text.strip() in IGNORE_UTTERANCES) or (labels.strip() in IGNORE_UTTERANCES)):
            return []
        lines.append({
            'text': conv['conversations'][2 * turn_idx]['text'],
            'labels': conv['conversations'][2 * turn_idx + 1]['text']
        })

    # add context data only if bot does not start conversations
    # In bot starting run we didn't use personas
    if not bot_start_conv:
        if ('bot_persona' in conv) and lines:
            if conv['bot_persona'].strip() != '':
                persona_text = 'your persona: ' + conv['bot_persona']
                lines[0]['text'] = persona_text + '\n' + lines[0]['text']
        elif ('context' in conv) and lines:
            persona_sentences = []
            if conv['context'].get('conv_theme') and conv['context'].get('conv_theme').get('theme_sentence'):
                persona_sentences.append('your persona: ' + conv['context']['conv_theme']['theme_sentence'])

            if conv['context'].get('personas') and conv.get('bot_role'):
                persona_sentences.append('your persona: ' + (conv['context']['personas']['robot_persona']
                                                             if conv.get('bot_role') == 'KARU'
                                                             else conv['context']['personas']['child_persona']))
            elif conv['context'].get('personas') and conv.get('worker_role'):
                persona_sentences.append('your persona: ' + (conv['context']['personas']['robot_persona']
                                                             if conv.get('worker_role') == 'CHILD'
                                                             else conv['context']['personas']['child_persona']))
            lines[0]['text'] = '\n'.join(persona_sentences) + '\n' + lines[0]['text']
        else:
            lines = []

    if lines:
        lines[-1]['episode_done'] = "True"

    text_lines = [dict_to_line(c) for c in lines]
    return text_lines


def dict_to_line(single_turn_conv):
    return '\t'.join([f'{key}:{_escape(value)}' for key, value in single_turn_conv.items()])


def _escape(value: str) -> str:
    return value.replace('\t', '\\t').replace('\n', '\\n').replace('|', '__PIPE__')
