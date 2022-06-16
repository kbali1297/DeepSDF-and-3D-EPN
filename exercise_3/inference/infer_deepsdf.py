import random
from pathlib import Path

import torch

from exercise_3.data.shape_implicit import ShapeImplicit
from exercise_3.model.deepsdf import DeepSDFDecoder
from exercise_3.util.misc import evaluate_model_on_grid


class InferenceHandlerDeepSDF:

    def __init__(self, latent_code_length, experiment, device):
        """
        :param latent_code_length: latent code length for the trained DeepSDF model
        :param experiment: path to experiment folder for the trained model; should contain "model_best.ckpt" and "latent_best.ckpt"
        :param device: torch device where inference is run
        """
        self.latent_code_length = latent_code_length
        self.experiment = Path(experiment)
        self.device = device
        self.truncation_distance = 0.01
        self.num_samples = 4096

    def get_model(self):
        """
        :return: trained deep sdf model loaded from disk
        """
        model = DeepSDFDecoder(self.latent_code_length)
        model.load_state_dict(torch.load(self.experiment / "model_best.ckpt", map_location='cpu'))
        model.eval()
        model.to(self.device)
        return model

    def get_latent_codes(self):
        """
        :return: latent codes which were optimized during training
        """
        latent_codes = torch.nn.Embedding.from_pretrained(torch.load(self.experiment / "latent_best.ckpt", map_location='cpu')['weight'])
        latent_codes.to(self.device)
        return latent_codes

    def reconstruct(self, points, sdf, num_optimization_iters):
        """
        Reconstructs by optimizing a latent code that best represents the input sdf observations
        :param points: all observed points for the shape which needs to be reconstructed
        :param sdf: all observed sdf values corresponding to the points
        :param num_optimization_iters: optimization is performed for this many number of iterations
        :return: tuple with mesh representations of the reconstruction
        """

        model = self.get_model()

        # TODO: define loss criterion for optimization
        loss_l1 = torch.nn.L1Loss()

        # initialize the latent vector that will be optimized
        latent = torch.ones(1, self.latent_code_length).normal_(mean=0, std=0.01).to(self.device)
        latent.requires_grad = True

        # TODO: create optimizer on latent, use a learning rate of 0.005
        optimizer = torch.optim.Adam([latent], lr = 0.005)

        for iter_idx in range(num_optimization_iters):
            # TODO: zero out gradients
            optimizer.zero_grad()
            # TODO: sample a random batch from the observations, batch size = self.num_samples
            batch_indices = random.sample(range(points.shape[0]), self.num_samples)

            batch_points = points[batch_indices, :]
            batch_sdf = sdf[batch_indices, :]

            # move batch to device
            batch_points = batch_points.to(self.device)
            batch_sdf = batch_sdf.to(self.device)

            # same latent code is used per point, therefore expand it to have same length as batch points
            latent_codes = latent.expand(self.num_samples, -1)

            # TODO: forward pass with latent_codes and batch_points
            predicted_sdf = model(torch.cat([latent_codes, batch_points], axis=1))

            # TODO: truncate predicted sdf between -0.1, 0.1
            predicted_sdf = torch.clamp(predicted_sdf, -0.1, 0.1)

            # compute loss wrt to observed sdf
            loss = loss_l1(predicted_sdf, batch_sdf)

            # regularize latent code
            loss += 1e-4 * torch.mean(latent.pow(2))

            # TODO: backwards and step
            loss.backward()

            optimizer.step()
            # loss logging
            if iter_idx % 50 == 0:
                print(f'[{iter_idx:05d}] optim_loss: {loss.cpu().item():.6f}')

        print('Optimization complete.')

        # visualize the reconstructed shape
        vertices, faces = evaluate_model_on_grid(model, latent.squeeze(0), self.device, 64, None)
        return vertices, faces

    def interpolate(self, shape_0_id, shape_1_id, num_interpolation_steps):
        """
        Interpolates latent codes between provided shapes and exports the intermediate reconstructions
        :param shape_0_id: first shape identifier
        :param shape_1_id: second shape identifier
        :param num_interpolation_steps: number of intermediate interpolated points
        :return: None, saves the interpolated shapes to disk
        """

        # get saved model and latent codes
        model = self.get_model()
        train_latent_codes = self.get_latent_codes()

        # get indices of shape_ids latent codes
        train_items = ShapeImplicit(4096, "train").items
        latent_code_indices = torch.LongTensor([train_items.index(shape_0_id), train_items.index(shape_1_id)]).to(self.device)

        # get latent codes for provided shape ids
        latent_codes = train_latent_codes(latent_code_indices)
        print(latent_codes.shape)
        for i in range(0, num_interpolation_steps + 1):
            # TODO: interpolate the latent codes: latent_codes[0, :] and latent_codes[1, :]
            interpolated_code = latent_codes[0,:] + i * (latent_codes[1, :] - latent_codes[0,:])/num_interpolation_steps
            # reconstruct the shape at the interpolated latent code
            evaluate_model_on_grid(model, interpolated_code, self.device, 64, self.experiment / "interpolation" / f"{i:05d}_000.obj")

    def infer_from_latent_code(self, latent_code_index):
        """
        Reconstruct shape from a given latent code index
        :param latent_code_index: shape index for a shape in the train set for which reconstruction is performed
        :return: tuple with mesh representations of the reconstruction
        """

        # get saved model and latent codes
        model = self.get_model()
        train_latent_codes = self.get_latent_codes()

        # get latent code at given index
        latent_code_indices = torch.LongTensor([latent_code_index]).to(self.device)
        latent_codes = train_latent_codes(latent_code_indices)

        # reconstruct the shape at latent code
        vertices, faces = evaluate_model_on_grid(model, latent_codes[0], self.device, 64, None)

        return vertices, faces

