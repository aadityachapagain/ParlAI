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
}


def build(opt):
    dpath = os.path.join(opt['datapath'], 'child_companion_dialog')
    version = opt.get('task_data_version', 'v0.0')

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES.get(version, RESOURCES['All']):
            downloadable_file.download_file(dpath)

        _create_parlai_format(dpath)

        build_data.mark_done(dpath, version_string=version)


def _create_parlai_format(dpath):
    conv_files = glob.glob(os.path.join(dpath, '*/*/t_*/custom/data.json'))
    conv_data = []
    for conv_file in conv_files:
        with open(conv_file, 'r') as f_read:
            data = json.load(f_read)
            if data['conversations']:
                conv_data.append(data)

    with open(os.path.join(dpath, 'train.txt'), 'w') as f_write:
        for conv in conv_data:
            for line in _get_lines(conv):
                f_write.write(f'{line} \n')


def _get_lines(conv):
    lines = []
    num_of_turns = len(conv['conversations']) // 2
    for turn_idx in range(num_of_turns):
        lines.append({
            'text': conv['conversations'][2 * turn_idx]['text'],
            'labels': conv['conversations'][2 * turn_idx + 1]['text']
        })

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
