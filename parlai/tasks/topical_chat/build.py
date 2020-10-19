import os
import glob
import json

from parlai.core import build_data
from parlai.utils.io import PathManager


RESOURCES = [
    build_data.DownloadableFile(
        'https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/train.json',
        'train.json',
        '39a079d8464ea5f44f0fd78dbc85fd4eed96303347cbf4f9301f823e3c0f437f',
        zipped=False,
    ),
    build_data.DownloadableFile(
        'https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/valid_freq.json',
        'valid_freq.json',
        'd14e8bde94fa5556d4aad8fd3fb666df674e4fef8b4193c8c3c06c72822a366b',
        zipped=False
    ),
    build_data.DownloadableFile(
        'https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/valid_rare.json',
        'valid_rare.json',
        'b0f3c59584ea4ed61e368dc94feade6ba499a207a6437573b028d1f3ff918714',
        zipped=False
    ),
    build_data.DownloadableFile(
        'https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/test_freq.json',
        'test_freq.json',
        '3c65693d59e40a1fb58ead35dcfe85fbcc88ea93a4d74557df620996f72c8f08',
        zipped=False
    ),
    build_data.DownloadableFile(
        'https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/test_rare.json',
        'test_rare.json',
        '4f21a18be16e310840280678e7bb4646852573b239b69183ddf8898f40119567',
        zipped=False
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'topical_chat')
    version = opt.get('task_data_version', 'v0.0')

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Format it for use with ParlAIDialogTeacher
        _create_parlai_format(dpath)

        build_data.mark_done(dpath, version_string=version)


def _create_parlai_format(dpath):
    data_dtypes = ['train', 'valid', 'test']
    for data_type in data_dtypes:
        load_paths = glob.glob(os.path.join(dpath, f'{data_type}*.json'))
        save_path = os.path.join(dpath, f'{data_type}.txt')

        data = {}
        for path in load_paths:
            print(f'Loading {path}....')
            with PathManager.open(path, 'r', encoding='utf8') as f_read:
                data.update(json.load(f_read))

        print(f'Saving to {save_path}.....')
        with PathManager.open(save_path, 'w', encoding='utf8') as f_write:
            for _, ep_data in data.items():
                for line in _get_lines(ep_data['content']):
                    f_write.write(f'{line} \n')


def _get_lines(conv):
    lines = []
    num_of_turns = len(conv) // 2

    for turn_idx in range(num_of_turns):
        lines.append({
            'text': conv[2 * turn_idx]['message'],
            'labels': conv[2 * turn_idx + 1]['message']
        })

    if lines:
        lines[-1]['episode_done'] = "True"

    text_lines = [dict_to_line(c) for c in lines]
    return text_lines


def dict_to_line(single_turn_conv):
    return '\t'.join([f'{key}:{_escape(value)}' for key, value in single_turn_conv.items()])


def _escape(value: str) -> str:
    return value.replace('\t', '\\t').replace('\n', '\\n').replace('|', '__PIPE__')
