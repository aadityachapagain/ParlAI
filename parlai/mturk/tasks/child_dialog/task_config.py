#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Any

task_config: Dict[str, Any] = {}

task_config['frontend_version'] = 0

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Chat with Child Companion Robot as a Child.'

"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config[
    'hit_description'] = (
    '5 minute task where you will play the role of a child aged 5-10 and '
    'chat with a child companion robot. '
    'You should aim for a conversation that is typical of children. Workers accepting '
    'HIT for the first time need to pass a qualification test.'
)

"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog,conversation,text,child,robot'

"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config[
    'task_description'
] = '''
<b><h4>Task Description</h4></b>
<br>
<b>
You will play the role of a child aged 5 to 10 talking with a child companion robot named Moxie.
</b>
<br>
Moxie is a robot from GRL and it communicates in a tone that is comparable to an eight year old child.
<br>
Moxie is a robot on a mission to understand how to be a good friend to humans.
<br>
<b>Chat with Moxie naturally and try to get to know each other, i.e. 
both ask questions and answer questions. Your response should be as specific as possible and 
match the length of other party.</b>
<br>
<br>
While having conversation, please also check the checkboxes before utterances made by Moxie that does not suit its 
characteristics. We call such utterances adult utterances. For example you can mark utterances like 
<br>
<i>"I'm a manager in IT, it was a lot at first but now I feel like I got used to it, they do not pay me enough"
<br>"I'm in my early twenties. I've been working here for a few years now."</i>
<br> as adult utterances.
'''
