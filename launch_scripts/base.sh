# 1) Finetune on all datasets
# 2) Eval single task
# 3) Task addition

# Run finetune.py
python finetune.py \
--data-location=../datasets/ \
--save=../results/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0

# Run eval_single_task.py
python eval_single_task.py \
--data-location=../datasets/ \
--save=../results/

# Run eval_task_addition.py
python eval_task_addition.py \
--data-location=../datasets/ \
--save=../results/