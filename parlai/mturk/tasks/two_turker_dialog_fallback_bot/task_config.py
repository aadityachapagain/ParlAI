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
task_config['hit_title'] = 'Chat playing a role of child or a robot.'

"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config[
    'hit_description'] = (
    'You will play either the role of a child or the robot Karu. '
    'You should aim for a conversation that is typical of children.'
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
You will play either the role of a child or the robot Karu.
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
<li><b><span style="color:red">Please try to be as specific as possible. For example, when talking about movies, talk about specific movies, characters and plots.</b></span></li>
<li><b><span style="color:red">No racism, sexism or otherwise offensive messages, or the submission will be rejected.</b></span></li>
</ul>
'''
