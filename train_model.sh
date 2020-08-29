export GOOGLE_APPLICATION_CREDENTIALS="gcp/fusemachineschat.json"
export PYTHONFAULTHANDLER=1
python parlai/distillation/distill_model.py \
--config-path parlai/distillation/distill_config.yml \
-t reddit_datasets \
-dt train:stream \
--model distillation/generator \
--init-model zoo:blender/blender_3B/model \
--dict-file zoo:blender/blender_3B/model.dict \
--skip-generation True \
--model-parallel True \
--lr-scheduler reduceonplateau \
--lr-scheduler-patience 4 \
--load-from-checkpoint True \
--run-tag custom_blenderbot_1 \
--fp16-impl mem_efficient \
--warmup_updates 100 \
--log_every_n_secs 10 \
--history-add-global-end-token end \
--fp16 True --text-truncate 128 --truncate 128 \
--label-truncate 128 --dict-tokenizer bytelevelbpe \
--dict-lower True -lr 5e-06 --optimizer adam \
--lr-scheduler reduceonplateau --gradient-clip 0.1 \
-veps 0.25 --betas 0.9,0.999 --update-freq 1 \
--batchsize 16 -vp 10 -vmt ppl -vmm min \
--save-after-valid True \
--student-model-file /tmp/model_file/custom_blender_1 \
--init-model-student /tmp/model_file/custom_blender_1.checkpoint \
-tblog True