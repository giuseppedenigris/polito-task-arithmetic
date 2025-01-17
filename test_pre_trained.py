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
None


def compute_accuracy(model: ImageClassifier, split: torch.utils.data.DataLoader, device) -> float:
    model.train(False)                      # Set model to evaluation mode
    model.to(device)                        # Move to GPU if device is cuda
    
    with torch.no_grad():
        corrects, total = 0, 0
        for batch in tqdm(split):
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

def compute_accuracies(model, val_split, test_split, device) -> dict:
    """Computes the accuracy of the model on the validation split and the test split"""
    
    # The result is a dict with the keys: ['train', 'test']     ('train' represents the validation split)
    result = {}

    print("Computing accuracy on Validation split")
    result['train'] = compute_accuracy(model, val_split, device)
    
    print("Computing accuracy on Test split")
    result['test'] = compute_accuracy(model, test_split, device)

    return result
    

if __name__ == '__main__':
    # Useful to see if the system supports cuda acceleration
    print("[INFO] Cuda acceleration:", "ON" if torch.cuda.is_available() else "OFF")

    # Get the cli arguments
    args = parse_arguments()

    # Each dataset represents a different downstream task for the model
    dataset_names = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # Load the pre-trained encoder
    pt_encoder = ImageEncoder(args)

    # Accuracies (on both val and test split) for pre-trained model
    results_pt = {}

    # Iterate over each dataset
    for dataset_name in dataset_names[:1]:
        # Get the classification head of the dataset
        head = get_classification_head(args, dataset_name + "Val")  # Get the open-vocabulary classifier of the dataset
        
        # Attach the classification head to the encoders
        pt_model = ImageClassifier(pt_encoder, head)

        # Obtain the Validation split of the dataset
        val_dataset = get_dataset(dataset_name + "Val", preprocess=pt_model.val_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=2)
        val_split = get_dataloader(val_dataset, is_train=False, args=args)

        # Obtain the Test split of the dataset
        test_dataset = get_dataset(dataset_name, preprocess=pt_model.val_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=2)
        test_split = get_dataloader(test_dataset, is_train=False, args=args)

        print("Collecting results on " + dataset_name + ", Pre-Trained model")
        results_pt[dataset_name] = compute_accuracies(pt_model, val_split, test_split, args.device)

        print("##" + dataset_name + ":", results_pt[dataset_name])
    
    with open(args.save + "test_results_pt.json", "w+") as fp:
        json.dump(results_pt, fp)
