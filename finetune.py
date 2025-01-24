## Global imports ##
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

## Local imports ##
from args import parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
import utils

## Static parameters ##
LOG_FREQUENCY = 50
DL_NUM_WORKERS = 4                          # Dataloader 'num_workers' parameter


class TrainingHistory:
    def __init__(self, metrics: list[str], criteria=[]):
        """Keeps track of the training history of the network, namely the accuracy and the loss on both the training and validation sets, and the logTrFIM. Stores the best network parameters according to the specified criteria"""
        self.metrics = metrics
        self.history = {metric: [] for metric in self.metrics}

        self.criteria = criteria
        for metric in criteria:
            assert metric in self.history.keys(), f"Chosen metric ({metric}) does not exist in this TrainingHistory object"

        self.best_metrics = {}
        self.best_params = {}

    def update(self, net, *args):
        """Updates the training history. Call this at each epoch"""
        assert len(args) == len(self.history.keys())
        for metric, value in zip(self.metrics, args):
            self.history[metric].append(value)

        for metric in self.criteria:
            if len(self.best_metrics) < len(self.criteria) or self.history[metric][-1] > self.best_metrics[metric]:
                self.best_metrics[metric] = self.history[metric][-1]
                self.best_params[metric] = {key: value.clone() for key, value in net.state_dict().items()}
                print(f"Epoch {len(self.history[metric])} | new model with higher {metric} (={self.best_metrics[metric]}) saved")

def compute_accuracy_and_loss(model, split: DataLoader, device: str, use_tqdm=True):
    model.train(False)                      # Set model to evaluation mode
    model.to(device)                        # Move to GPU if device is cuda
    
    corrects, loss, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(split) if use_tqdm else split:
            # Bring data over the device of choice
            data = maybe_dictionarize(batch)
            images, labels = data["images"].to(device), data["labels"].to(device)

            # Forward Pass
            outputs = model(images)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update corrects, loss and total
            corrects += torch.sum(preds == labels.data).data.item()
            loss += criterion(outputs, labels).item() * labels.size(0)
            total += len(images)

    # Calculate Accuracy and normalize loss
    accuracy = corrects / total
    loss /= total
    return accuracy, loss


if __name__ == '__main__':
    # Useful to see if the system supports cuda acceleration
    print("[INFO] Cuda acceleration:", "ON" if torch.cuda.is_available() else "OFF")

    # Get the cli arguments
    args = parse_arguments()

    # Print the chosen parameters
    print( "[INFO] Finetuning parameters")
    print(f"       >   batch size: {args.batch_size}")
    print(f"       >   learning rate: {args.lr}")
    print(f"       >   weight decay: {args.wd}")

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()                         # Loss function

    # Each dataset represents a different downstream task for the model
    dataset_names = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # Each dataset should be trained with this number of epochs to balance the total number of iterations across the different datasets
    dataset_epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}
    
    # Instantiate a full model architecture
    encoder = ImageEncoder(args)                                    # Pre-trained CLIP ViT backbone

    # # Save pre-trained weights (don’t need to store classification heads)
    # encoder.save(args.save + "results_logTrFIM/encoder_Zeroshot.pt")
    # encoder.save(args.save + "results_val_accuracy/encoder_Zeroshot.pt")

    # Save pre-trained weights (don’t need to store classification heads)
    encoder.save(args.save + "encoder_Zeroshot.pt")

    # Iterate over each dataset
    for dataset_name in dataset_names:
        print("Starting finetuning process on " + dataset_name)
        # Instantiate a full model architecture
        encoder = ImageEncoder(args)                                # Pre-trained CLIP ViT backbone
        
        # We use SGD without momentum (or with?)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.wd)

        # Attach the classification head to the encoder
        head = get_classification_head(args, dataset_name + "Val")  # Get the open-vocabulary classifier of the dataset
        model = ImageClassifier(encoder, head)                      # Build full model
        model.freeze_head()                                         # Freeze the classification head

        # Move to GPU if device is cuda
        model.to(args.device)

        # Obtain the Train split of the dataset
        train_dataset = get_dataset(dataset_name + "Val", preprocess=model.train_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=DL_NUM_WORKERS)
        train_split = get_dataloader(train_dataset, is_train=True, args=args)
        
        # Obtain the Validation split of the dataset
        val_dataset = get_dataset(dataset_name + "Val", preprocess=model.val_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=DL_NUM_WORKERS)
        val_split = get_dataloader(val_dataset, is_train=False, args=args)
        
        # Use a TrainingHistory object to track the best model based on logTrFIM and Validation Accuracy
        finetune_history = TrainingHistory(["train_accuracy", "train_loss", "val_accuracy", "val_loss", "logTrFIM"], criteria=["logTrFIM", "val_accuracy"])

        # Start iterating over the epochs
        current_step = 0
        for epoch in range(dataset_epochs[dataset_name]):
            print('Starting epoch {}/{}'.format(epoch+1, dataset_epochs[dataset_name]))

            # Iterate over the dataset
            for batch in train_split:
                # Bring data over the device of choice
                data = maybe_dictionarize(batch)
                images, labels = data["images"].to(args.device), data["labels"].to(args.device)

                model.train()                                       # Sets module in training mode

                optimizer.zero_grad()                               # Zero-ing the gradients

                # Forward pass to the network
                outputs = model(images)

                # Compute loss based on output and ground truth
                loss = criterion(outputs, labels)

                # Log loss
                if current_step % LOG_FREQUENCY == 0:
                    print('Step {}, Loss {}'.format(current_step, loss.item()))

                # Compute gradients and update weights
                loss.backward()                                     # backward pass: computes gradients
                optimizer.step()                                    # update weights based on accumulated gradients

                current_step += 1
            
            # print(f"Epoch {epoch+1} | collecting results on Train split")
            # train_accuracy, train_loss = compute_accuracy_and_loss(model, train_split, args.device)
            
            # print(f"Epoch {epoch+1} | collecting results on Validation split")
            # val_accuracy, val_loss = compute_accuracy_and_loss(model, val_split, args.device)

            # print(f"Epoch {epoch+1} | computing logTrFIM")
            # logTrFIM = utils.train_diag_fim_logtr(args, model, dataset_name, samples_nr=200)

            # finetune_history.update(model.image_encoder, train_accuracy, train_loss, val_accuracy, val_loss, logTrFIM)


        model.image_encoder.save(args.save + "encoder_" + dataset_name + ".pt")

        # Save the finetune history to file
        # print("Saving finetune history of " + dataset_name)
        # with open(args.save + "ft_history_" + dataset_name + ".json", "w") as fp:
        #     json.dump(finetune_history.history, fp)

        # Save best encoder based on logTrFIM
        # model.image_encoder.load_state_dict(finetune_history.best_params["logTrFIM"])
        # model.image_encoder.save(args.save + "results_logTrFIM/encoder_" + dataset_name + ".pt")

        ### ------
        # acc, loss = compute_accuracy_and_loss(model, val_split, args.device, use_tqdm=False)
        # print(f"val | acc: {acc} loss: {loss}")
        ### ------

        # # Save best encoder based on Validation Accuracy
        # model.image_encoder.load_state_dict(finetune_history.best_params["val_accuracy"])
        # model.image_encoder.save(args.save + "results_val_accuracy/encoder_" + dataset_name + ".pt")
        
        # ### ------
        # acc, loss = compute_accuracy_and_loss(model, val_split, args.device, use_tqdm=False)
        # print(f"val | acc: {acc} loss: {loss}")
        # ### ------
