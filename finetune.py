## Global imports ##
import torch

## Local imports ##
from args import parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head

## Static parameters ##
LOG_FREQUENCY = 50

if __name__ == '__main__':
    # Useful to see if the system supports cuda acceleration
    print("[INFO] Cuda acceleration:", "ON" if torch.cuda.is_available() else "OFF")

    # Get the cli arguments
    args = parse_arguments()

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()                         # Loss function

    # Each dataset represents a different downstream task for the model
    dataset_names = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # Each dataset should be trained with this number of epochs to balance the total number of iterations across the different datasets
    dataset_epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}
    
    # Instantiate a full model architecture
    encoder = ImageEncoder(args)                                    # Pre-trained CLIP ViT backbone

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
        train_dataset = get_dataset(dataset_name + "Val", preprocess=model.train_preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=2)
        train_split = get_dataloader(train_dataset, is_train=True, args=args)

        # Start iterating over the epochs
        current_step = 0
        for epoch in range(dataset_epochs[dataset_name]):
            print('Starting epoch {}/{}'.format(epoch+1, dataset_epochs[dataset_name]))

            # Iterate over the dataset
            for batch in train_split:
                # Bring data over the device of choice
                data = maybe_dictionarize(batch)
                images, labels = data["images"].to(args.device), data["labels"].to(args.device)

                model.train() # Sets module in training mode

                optimizer.zero_grad() # Zero-ing the gradients

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

        # Save fine-tuned weights (don’t need to store classification heads)
        model.image_encoder.save(args.save + "encoder_" + dataset_name + ".pt")