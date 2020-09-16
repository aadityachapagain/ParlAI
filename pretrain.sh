export GOOGLE_APPLICATION_CREDENTIALS="gcp/fusemachineschat.json"

export WANDB_API_KEY="250ec687322ac7425ad946e7c5238228d48fc93a"

python3 parlai/tasks/reddit_datasets/build.py

python parlai/scripts/multiprocessing_train.py \
-t reddit_datasets \
-dt train:stream \
--run-tag "90m_pretrain" \
--wand-project-name "Karu_chatbot_v0" \
--wand-run-name "90M Model Pretraining" \
--wand-id "90MmodelPretraining0" \
-m transformer/generator \
--load-from-checkpoint True \
--embedding-size 512 --n-layers 8 --ffn-size 2048 \
--dropout 0.1 --n-heads 16 --learn-positional-embeddings True \
--n-positions 512 --variant xlm \
--activation gelu --skip-generation True --fp16 True \
--text-truncate 256 --label-truncate 128 --dict-tokenizer bytelevelbpe \
-lr 2e-06 --optimizer adam --lr-scheduler invsqrt \
--lr-scheduler-patience 3 \
--warmup_updates 3125 \
--gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 2 \
--log_every_n_secs 30 \
--attention-dropout 0.0 --relu-dropout 0.0 \
--skip-generation True -vp 15 -stim 60 \
-vme 20000 -vmt ppl -vmm min \
--save-after-valid True \
--save-every-n-secs 1800 \
--dict-file zoo:blender/blender_3B/model.dict \
--delimiter '  ' \
--fp16-impl apex \
--dynamic-batching full --batchsize 56 --eval-batchsize 64 \
--model-file /tmp/models/Karu/karu_bot_90M \
--init-model /tmp/models/Karu/karu_bot_90M.checkpoint \
-tblog True \
--validation-patience 10 \
--inference beam  --beam-size 10 --beam-block-ngram 3 \
--beam-context-block-ngram 3 \
--beam-min-length 20 \

#--invsqrt-lr-decay-gamma 0.01 \