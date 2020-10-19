python parlai/scripts/interactive_self_chat_beams.py -mf zoo:blender/blender_3B/model --inference delayedbeam --recursion 2 --adultlike_file_path data/adult_like_statements_sampled.txt

python parlai/scripts/interactive_self_chat_beams.py -mf zoo:blender/blender_3B/model --inference beam --recursion 2 --adultlike_file_path data/adult_like_statements_sampled.txt

python parlai/scripts/interactive_self_chat_beams.py -mf zoo:blender/blender_3B/model --inference topk --recursion 2 --adultlike_file_path data/adult_like_statements_sampled.txt

python parlai/scripts/interactive_self_chat_beams.py -mf zoo:blender/blender_3B/model --inference nucleus --recursion 2 --adultlike_file_path data/adult_like_statements_sampled.txt