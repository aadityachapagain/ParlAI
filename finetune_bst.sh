export GOOGLE_APPLICATION_CREDENTIALS="gcp/fusemachineschat.json"
export PYTHONFAULTHANDLER=1
python parlai/scripts/train_model.py \
-t blended_skill_talk,wizard_of_wikipedia,convai2:normalized \
-m transformer/generator --multitask-weights 1,3,3,3 \
--embedding-size 512 --n-layers 8 --ffn-size 2048 \
--dropout 0.1 --n-heads 16 --learn-positional-embeddings True \
--n-positions 512 --variant prelayernorm \
--activation gelu --skip-generation True --fp16 True \
--text-truncate 512 --label-truncate 128 --dict-tokenizer bytelevelbpe \
--dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau \
--gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 2 \
--attention-dropout 0.0 --relu-dropout 0.0 \
--skip-generation True -vp 15 -stim 60 \
-vme 20000 -bs 16 -vmt ppl -vmm min \
--save-after-valid True \
--delimiter '  ' \
--fp16-impl apex \
--dynamic-batching full --batchsize 32 \
--model-file data/models/Karu/karu_bot_90M \
--init-model data/models/Karu/karu_bot_90M.checkpoint \