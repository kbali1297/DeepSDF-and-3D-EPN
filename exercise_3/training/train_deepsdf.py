from pathlib import Path

import torch

from exercise_3.model.deepsdf import DeepSDFDecoder
from exercise_3.data.shape_implicit import ShapeImplicit
from exercise_3.util.misc import evaluate_model_on_grid


def train(model, latent_vectors, train_dataloader, device, config):

    # Declare loss and move to device
    # TODO: declare loss as `loss_criterion`
    loss_criterion = torch.nn.L1Loss()
    loss_criterion.to(device)

    # declare optimizer
    optimizer = torch.optim.Adam([
        {
            # TODO: optimizer params and learning rate for model (lr provided in config)
            'params':model.parameters(), 'lr':config['learning_rate_model']
        },
        {
            # TODO: optimizer params and learning rate for latent code (lr provided in config)
            'params':latent_vectors.parameters(), 'lr':config['learning_rate_code']
        }
    ])

    # declare learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # Set model to train
    model.train()

    # Keep track of running average of train loss for printing
    train_loss_running = 0.

    # Keep track of best training loss for saving the model
    best_loss = float('inf')

    for epoch in range(config['max_epochs']):

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            ShapeImplicit.move_batch_to_device(batch, device)

            # TODO: Zero out previously accumulated gradients
            optimizer.zero_grad()
            # calculate number of samples per batch (= number of shapes in batch * number of points per shape)
            num_points_per_batch = batch['points'].shape[0] * batch['points'].shape[1]

            # get latent codes corresponding to batch shapes
            # expand so that we have an appropriate latent vector per sdf sample
            batch_latent_vectors = latent_vectors(batch['indices']).unsqueeze(1).expand(-1, batch['points'].shape[1], -1)
            batch_latent_vectors = batch_latent_vectors.reshape((num_points_per_batch, config['latent_code_length']))

            # reshape points and sdf for forward pass
            points = batch['points'].reshape((num_points_per_batch, 3))
            sdf = batch['sdf'].reshape((num_points_per_batch, 1))

            # TODO: perform forward pass
            predicted_sdf = model(torch.cat([batch_latent_vectors, points], axis=1))
            # TODO: truncate predicted sdf between -0.1 and 0.1
            predicted_sdf = torch.clamp(predicted_sdf, -0.1, 0.1)

            # compute loss
            loss = loss_criterion(predicted_sdf, sdf)

            # regularize latent codes
            code_regularization = torch.mean(torch.norm(batch_latent_vectors, dim=1)) * config['lambda_code_regularization']
            if epoch > 100:
                loss = loss + code_regularization

            # TODO: backward
            loss.backward()
            # TODO: update network parameters
            optimizer.step()
            # loss logging
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                train_loss = train_loss_running / config["print_every_n"]
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss:.6f}')

                # save best train model and latent codes
                if train_loss < best_loss:
                    torch.save(model.state_dict(), f'exercise_3/runs/{config["experiment_name"]}/model_best.ckpt')
                    torch.save(latent_vectors.state_dict(), f'exercise_3/runs/{config["experiment_name"]}/latent_best.ckpt')
                    best_loss = train_loss

                train_loss_running = 0.

            # visualize first 5 training shape reconstructions from latent codes
            if iteration % config['visualize_every_n'] == (config['visualize_every_n'] - 1):
                # Set model to eval
                model.eval()
                latent_vectors_for_vis = latent_vectors(torch.LongTensor(range(min(5, latent_vectors.num_embeddings))).to(device))
                for latent_idx in range(latent_vectors_for_vis.shape[0]):
                    # create mesh and save to disk
                    evaluate_model_on_grid(model, latent_vectors_for_vis[latent_idx, :], device, 64, f'exercise_3/runs/{config["experiment_name"]}/meshes/{iteration:05d}_{latent_idx:03d}.obj')
                # set model back to train
                model.train()

        # lr scheduler update
        scheduler.step()


def main(config):
    """
    Function for training DeepSDF
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'num_sample_points': number of sdf samples per shape while training
                   'latent_code_length': length of deepsdf latent vector
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate_model': learning rate of model optimizer
                   'learning_rate_code': learning rate of latent code optimizer
                   'lambda_code_regularization': latent code regularization loss coefficient
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'visualize_every_n': visualize some training shapes every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # create dataloaders
    train_dataset = ShapeImplicit(config['num_sample_points'], 'train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=0,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # Instantiate model
    model = DeepSDFDecoder(config['latent_code_length'])
    # Instantiate latent vectors for each training shape
    latent_vectors = torch.nn.Embedding(len(train_dataset), config['latent_code_length'], max_norm=1.0)

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'] + "_model.ckpt", map_location='cpu'))
        latent_vectors = torch.nn.Embedding.from_pretrained(torch.load(config['resume_ckpt'] + "_latent.ckpt", map_location='cpu'))

    # Move model to specified device
    model.to(device)
    latent_vectors.to(device)

    # Create folder for saving checkpoints
    Path(f'exercise_3/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, latent_vectors, train_dataloader, device, config)
