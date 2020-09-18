python parlai/core/latest_checkpoint.py --model-file /tmp/models/talk_with_bot/Karu/karu_bot_90M --run-tag 90m_pretrain_2

python parlai/scripts/interactive.py --model-file /tmp/models/talk_with_bot/Karu/karu_bot_90M.checkpoint --model transformer/generator
