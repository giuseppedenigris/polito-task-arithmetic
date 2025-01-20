## Global imports ##
import torch
import json
from tqdm import tqdm

## Local imports ##
from args import parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader
from modeling import ImageClassifier
from heads import get_classification_head
import utils
import eval_single_task
from task_vectors import NonLinearTaskVector

## Static parameters ##
DL_NUM_WORKERS = 2                          # Dataloader 'num_workers' parameter


if __name__ == '__main__':
    # Useful to see if the system supports cuda acceleration
    print("[INFO] Cuda acceleration:", "ON" if torch.cuda.is_available() else "OFF")

    # Get the cli arguments
    args = parse_arguments()

    # Each dataset represents a different downstream task for the model
    dataset_names = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # Build task vectors
    task_vectors = {}
    for dataset_name in dataset_names:
        task_vectors[dataset_name] = NonLinearTaskVector(args.save + "encoder_Zeroshot.pt", args.save + "encoder_" + dataset_name + ".pt")

    # Add them together (without scaling)
    task_vector_sum = sum(task_vectors.values())

    # Load the classification heads
    classification_heads = {}
    for dataset_name in dataset_names:
        classification_heads[dataset_name] = get_classification_head(args, dataset_name + "Val")    # Get the open-vocabulary classifier of the dataset

    # Load single_task results of the finetuned models
    with open(args.save + "results_ft.json") as fp:
        st_results = json.load(fp)

    # Find optimal alpha -----------------------------------------------------------------------------------------------

    # Alpha is searched in the range [0, 1] with steps of 0.05
    alpha_range = [x/100 for x in range(5,101,5)]

    # Some variables where to cache Validation datasets and Validation dataloaders
    val_datasets, val_splits = {}, {}
    for dataset_name in dataset_names:
        val_datasets[dataset_name], val_splits[dataset_name] = None, None
    
    # Compute normalized accuracies for each alpha
    norm_accuracies = {}
    for alpha in tqdm(alpha_range):
        # Build the encoder with alpha scaling
        merged_encoder = task_vector_sum.apply_to(args.save + "encoder_Zeroshot.pt", scaling_coef=alpha)

        # Compute normalized accuracies
        norm_accuracies[alpha] = {}
        for dataset_name in dataset_names:
            # Build full classification model
            merged_model = ImageClassifier(merged_encoder, classification_heads[dataset_name])

            # Load Validation splits of the datasets if they're not cached yet
            if val_datasets[dataset_name] is None:
                val_datasets[dataset_name] = get_dataset(dataset_name + "Val", preprocess=merged_model.val_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=2)
                val_splits[dataset_name] = get_dataloader(val_datasets[dataset_name], is_train=False, args=args)

            # Compute absolute accuracy
            abs_accuracy = eval_single_task.compute_accuracy(merged_model, val_splits[dataset_name], args.device)

            # Normalize w.r.t. the single_task accuracy of the finetuned model
            norm_accuracies[alpha][dataset_name] = abs_accuracy/st_results[dataset_name]["train"]
        
        # Compute Average Normalized Accuracy
        norm_accuracies[alpha]["Average"] = sum(norm_accuracies[alpha].values()) / len(dataset_names)
    
    # Select alpha with maximum Avg Normalized Accuracy
    alpha = max(norm_accuracies, key=lambda alpha: norm_accuracies[alpha]["Average"])

    # --------------------------------------------------------------------------------------------------------------------

    # Compute metrics ----------------------------------------------------------------------------------------------------
    
    # Absolute Accuracies (on both train and test split) and logTrFIM for scaled finetuned models
    results_sf = {}

    # Absolute and Normalized Accuracies (on both train and test split) and logTrFIM for the merged model
    results_mg = {}

    # Build the Merged encoder
    mg_encoder = task_vector_sum.apply_to(args.save + "encoder_Zeroshot.pt", scaling_coef=alpha)

    # Iterate over each dataset
    for dataset_name in dataset_names:
        # Get the classification head of the dataset
        head = get_classification_head(args, dataset_name + "Val")      # Get the open-vocabulary classifier of the dataset

        # Build the Scaled Finetuned encoder
        sf_encoder = task_vectors[dataset_name].apply_to(args.save + "encoder_Zeroshot.pt", scaling_coef=alpha)
        
        # Attach the classification head to the encoders
        mg_model = ImageClassifier(mg_encoder, head)
        sf_model = ImageClassifier(sf_encoder, head)

        # Obtain the Train split of the dataset
        train_dataset = get_dataset(dataset_name + "Val", preprocess=mg_model.val_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=2)
        train_split = get_dataloader(train_dataset, is_train=True, args=args)

        # Obtain the Test split of the dataset
        test_dataset = get_dataset(dataset_name, preprocess=mg_model.val_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=DL_NUM_WORKERS)
        test_split = get_dataloader(test_dataset, is_train=False, args=args)

        print("### Collecting results on " + dataset_name + ", Scaled-Finetuned model")
        results_sf[dataset_name] = {}
        
        print("# Computing accuracy on Train split")
        results_sf[dataset_name]['train'] = eval_single_task.compute_accuracy(sf_model, train_split, args.device)
        print("# Computing accuracy on Test split")
        results_sf[dataset_name]['test'] = eval_single_task.compute_accuracy(sf_model, test_split, args.device)
        print("# Computing logTraceFIM")
        results_sf[dataset_name]['logTrFIM'] = utils.train_diag_fim_logtr(args, sf_model, dataset_name)
        

        print("### Collecting results on " + dataset_name + ", Merged model")
        results_mg[dataset_name] = {}
        
        print("# Computing accuracy on Train split")
        results_mg[dataset_name]['train Absolute'] = eval_single_task.compute_accuracy(mg_model, train_split, args.device)
        results_mg[dataset_name]['train Normalized'] = results_mg[dataset_name]['train Absolute'] / st_results[dataset_name]["train"]
        print("# Computing accuracy on Test split")
        results_mg[dataset_name]['test Absolute'] = eval_single_task.compute_accuracy(mg_model, test_split, args.device)
        results_mg[dataset_name]['test Normalized'] = results_mg[dataset_name]['test Absolute'] / st_results[dataset_name]["test"]
        print("# Computing logTraceFIM")
        results_mg[dataset_name]['logTrFIM'] = utils.train_diag_fim_logtr(args, mg_model, dataset_name)
        
    # --------------------------------------------------------------------------------------------------------------------

    with open(args.save + "results_sf.json", "w+") as fp:
        json.dump(results_sf, fp)
    
    with open(args.save + "results_mg.json", "w+") as fp:
        json.dump(results_mg, fp)