export GOOGLE_APPLICATION_CREDENTIALS="gcp/fusemachineschat.json"
export PYTHONFAULTHANDLER=1
python parlai/distillation/multiprocessing_distill.py \
--config-path parlai/distillation/distill_config.yml \
-t reddit_datasets \
-dt train:stream \
--model distillation/generator \
--init-model zoo:blender/blender_3B/model \
--dict-file zoo:blender/blender_3B/model.dict \
--skip-generation True \
--lr-scheduler reduceonplateau \
--lr-scheduler-patience 4 \
--load-from-checkpoint True \
--run-tag karu_bot_v0 \
--run-tag "karu_90M" \
--gcs-data-path  "reddit/20200903" \
--fp16-impl mem_efficient \
--warmup_updates 100 \
--wand-project-name "Karu_chatbot_v0" \
--wand-run-name "90M Model Distillation" \
--wand-id "90MmodelDistill" \
--log_every_n_secs 10 \
--history-add-global-end-token end \
--evaltask reddit_datasets --eval_batchsize 12 \
--fp16 True --text-truncate 128 --truncate 128 \
--label-truncate 128 --dict-tokenizer bytelevelbpe \
-lr 1e-06 --optimizer adamax \
--lr-scheduler reduceonplateau --gradient-clip 0.1 \
-veps 0.25 --betas 0.9,0.999 --update-freq 2 \
-vp 10 -vmt ppl -vmm min \
--dynamic-batching full --batchsize 4 \
--delimiter '  ' \
--save-after-valid True \
--student-model-file data/models/Karu/karu_bot_90M \
--init-model-student data/models/Karu/karu_bot_90M.checkpoint \
--save-every-n-secs 3600 \
-tblog True

# --validation-every-n-secs 14400 \
# --dynamic-batching full 