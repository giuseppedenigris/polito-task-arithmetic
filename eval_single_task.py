## Global imports ##
import torch
import json
from tqdm import tqdm

## Local imports ##
from args import parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
import utils

## Static parameters ##
DL_NUM_WORKERS = 2                          # Dataloader 'num_workers' parameter


def compute_accuracy(model: ImageClassifier, split: torch.utils.data.DataLoader, device: str, use_tqdm=True) -> float:
    model.train(False)                      # Set model to evaluation mode
    model.to(device)                        # Move to GPU if device is cuda
    
    with torch.no_grad():
        corrects, total = 0, 0
        for batch in tqdm(split) if use_tqdm else split:
            # Bring data over the device of choice
            data = maybe_dictionarize(batch)
            images, labels = data["images"].to(device), data["labels"].to(device)

            # Forward Pass
            outputs = model(images)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update corrects and total
            corrects += torch.sum(preds == labels.data).data.item()
            total += len(images)

    # Calculate Accuracy
    accuracy = corrects / float(total)
    return accuracy  

if __name__ == '__main__':
    # Useful to see if the system supports cuda acceleration
    print("[INFO] Cuda acceleration:", "ON" if torch.cuda.is_available() else "OFF")

    # Get the cli arguments
    args = parse_arguments()

    # Each dataset represents a different downstream task for the model
    dataset_names = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # Load the pre-trained encoder
    pt_encoder = utils.torch_load(args.save + "encoder_Zeroshot.pt", args.device)

    # Accuracies (on both train and test split) and logTrFIM for fine-tuned models
    results_ft = {}

    # Iterate over each dataset
    for dataset_name in dataset_names:
        # Get the classification head of the dataset
        head = get_classification_head(args, dataset_name + "Val")  # Get the open-vocabulary classifier of the dataset

        # Load the encoder finetuned on the dataset
        ft_encoder = utils.torch_load(args.save + "encoder_" + dataset_name + ".pt", args.device)
        
        # Attach the classification head to the encoders
        pt_model = ImageClassifier(pt_encoder, head)
        ft_model = ImageClassifier(ft_encoder, head)

        # Obtain the Train split of the dataset
        train_dataset = get_dataset(dataset_name + "Val", preprocess=pt_model.val_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=2)
        train_split = get_dataloader(train_dataset, is_train=True, args=args)

        # Obtain the Test split of the dataset
        test_dataset = get_dataset(dataset_name, preprocess=pt_model.val_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=DL_NUM_WORKERS)
        test_split = get_dataloader(test_dataset, is_train=False, args=args)
        
        print("### Collecting results on " + dataset_name + ", Fine-Tuned model")
        results_ft[dataset_name] = {}
        
        print("# Computing accuracy on Train split")
        results_ft[dataset_name]['train'] = compute_accuracy(ft_model, train_split, args.device)
        print("# Computing accuracy on Test split")
        results_ft[dataset_name]['test'] = compute_accuracy(ft_model, test_split, args.device)
        print("# Computing logTraceFIM")
        results_ft[dataset_name]['logTrFIM'] = utils.train_diag_fim_logtr(args, ft_model, dataset_name)

    
    with open(args.save + "results_ft.json", "w+") as fp:
        json.dump(results_ft, fp)
