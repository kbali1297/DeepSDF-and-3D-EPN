from pathlib import Path
from pickletools import optimize

import numpy as np
import torch

from src.model.threedepn import ThreeDEPN
from src.data.shapenet import ShapeNet


def train(model, train_dataloader, val_dataloader, device, config):
     # Declare loss and move to device; we need both smoothl1 and pure l1 losses here
    loss_criterion_test = torch.nn.L1Loss().to(device)
    loss_criterion_train = torch.nn.SmoothL1Loss().to(device)

     # Declare optimizer with learning rate given in config
    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['learning_rate'])
    # Here, we follow the original implementation to also use a learning rate scheduler -- it simply reduces the learning rate to half every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

     # Set model to train
    model.train()
    best_loss_val = np.inf

    # Keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
             # Move batch to device, set optimizer gradients to zero, perform forward pass
            ShapeNet.move_batch_to_device(batch, device)

            optimizer.zero_grad()
            reconstruction = model(batch['input_sdf'])
            # Mask out known regions -- only use loss on reconstructed, previously unknown regions
            reconstruction[batch['input_sdf'][:, 1] == 1] = 0  # Mask out known
            target = batch['target_df']
            target[batch['input_sdf'][:, 1] == 1] = 0

             # Compute loss, Compute gradients, Update network parameters

            loss = loss_criterion_train(reconstruction, target)

            loss.backward()

            optimizer.step()

            #loss = loss_criterion_train(reconstruction, target)
            # Logging
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
                train_loss_running = 0.

            # Validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                 # Set model to eval
                model.eval()
                # Evaluation on entire validation set
                loss_val = 0.
                for batch_val in val_dataloader:
                    ShapeNet.move_batch_to_device(batch_val, device)

                    with torch.no_grad():
                        reconstruction = model(batch_val['input_sdf'])

                        # Transform back to metric space
                        # We perform our validation with a pure l1 loss in metric space for better comparability
                        reconstruction = torch.exp(reconstruction) - 1
                        target = torch.exp(batch_val['target_df']) - 1
                        # Mask out known regions -- only report loss on reconstructed, previously unknown regions
                        reconstruction[batch_val['input_sdf'][:, 1] == 1] = 0
                        target[batch_val['input_sdf'][:, 1] == 1] = 0

                    loss_val += loss_criterion_test(reconstruction, target).item()

                loss_val /= len(val_dataloader)
                if loss_val < best_loss_val:
                    torch.save(model.state_dict(), f'src/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_loss_val = loss_val

                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')

                 # Set model back to train
                model.train()
        scheduler.step()


def main(config):
    """
    Function for training PointNet on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ShapeNet('train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=1,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=train_dataset.worker_init_fn  : Uncomment this line if  we are using shapenet_zip on Google Colab
    )

    val_dataset = ShapeNet('val' if not config['is_overfit'] else 'overfit')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=1,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=val_dataset.worker_init_fn  : Uncomment this line if  we are using shapenet_zip on Google Colab
    )

    # Instantiate model
    model = ThreeDEPN()

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'src/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)
