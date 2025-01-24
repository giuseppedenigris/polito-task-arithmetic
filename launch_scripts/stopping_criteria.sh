# 1) Finetune on all datasets
# 2) Eval single task
# 3) Task addition

# Run finetune.py which produces two results based on two different stopping criteria: "results/results_logTrFIM/" and "results/results_val_accuracy/" 
python finetune.py \
--data-location=../datasets/ \
--save=../results/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0

# Run eval_single_task.py for "results/results_logTrFIM/"
python eval_single_task.py \
--data-location=../datasets/ \
--save=../results/results_logTrFIM/

# Run eval_task_addition.py for "results/results_logTrFIM/"
python eval_task_addition.py \
--data-location=../datasets/ \
--save=../results/results_logTrFIM/

# Run eval_single_task.py for "results/results_val_accuracy/"
python eval_single_task.py \
--data-location=../datasets/ \
--save=../results/results_val_accuracy/

# Run eval_task_addition.py for "results/results_val_accuracy/"
python eval_task_addition.py \
--data-location=../datasets/ \
--save=../results/results_val_accuracy/
