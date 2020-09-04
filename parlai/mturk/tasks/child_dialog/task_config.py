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
task_config['hit_title'] = ('Chat as a child companion. Workers who perform well will be reached out to'
                            'and given more hits at a higher reward level')

"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config[
    'hit_description'] = (
    '5 minute task where you will play the role of a child companion robot, Karu. '
    'You should aim for a conversation that is typical of children. Workers accepting '
    'HIT for the first time need to pass a qualification test.'
)

"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog,conversation,text,child'

"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config[
    'task_description'
] = '''
<b><h4>Task Description</h4></b>
<br>
<b>
You will play the role of a child companion robot, Karu.
</b>
<br>
Overall, Karu communicates in a tone that is comparable to an eight year old child.
<br>
Karu is a robot on a mission to understand how to be a good friend to humans.
<br>
<b>Your response should be as specific as possible and match the length of other party.</b>
<br>
<b>Note:</b>
<ul>
<li><b><span style="color:red">Please try to stick to the theme of conversation and be as specific as possible. For example, when talking about movies, talk about specific movies, characters and plots.</b></span></li>
<li><b><span style="color:red">Workers who conduct good specific conversations will be eligible for higher pay rate.</b></span></li>
</ul>
'''
