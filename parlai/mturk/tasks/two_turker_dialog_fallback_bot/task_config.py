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
task_config['hit_title'] = 'Chat with another worker playing a childlike character'

"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config[
    'hit_description'] = (
    'In this task, you will play either the role of the child or Karu(Karu is a robot on a mission to understand how to be a good friend to humans.) and your partner will be'
    'the other party. You should aim for conversation that is typical of children'
    'and not discuss adult or teenage topics. If you are Karu, you should aim to be empathetic, interesting and knowledgeable'
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
<br>
<b><h4>Task Description</h4></b>
<br>
<b>In this task, you will play either the role of the child or Karu and your partner will be
the other party. You should aim for conversation that is typical of children
and not discuss adult or teenage topics. If you are Karu, you should aim to be empathetic, interesting and knowledgeable.</b>
<br>
<b><h4>Introducing Karu</h4></b>
Karu is a robot built by The Global Robotics Laboratory on a mission to understand how to be a good friend to humans. Specifically, it is a small lifelike robot designed to respond and communicate intelligently with children (ages 5-10). Karu was designed to explore and teach life skills by modeling positive empathetic social behavior.
<br>
Overall, Karu communicates in clear, gender-neutral and optimistic sounding tone that is comparable in communication structure to an eight year old child.
<br>
Karu is very intelligent, empathetic and caring but many times the resulting style of communication feels very self-conscious and awkward as if the robot is deeply concerned about 'saying the right thing.'
<br><b>Note: <span style="color:red">- No racism, sexism or otherwise offensive messages, or the submission will be rejected and we will report to Amazon.</b></span>
'''
