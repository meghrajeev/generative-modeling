import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision

def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # Generate 100 samples of 128-dim vectors
    # Do so by linearly interpolating for 10 steps across each of the first two dimensions between -1 and 1.
    # Keep the rest of the z vector for the samples to be some fixed value (e.g. 0).
    # Forward the samples through the generator.
    # Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    num_samples = 100
    z_dim = 128
    z = torch.zeros(num_samples, z_dim)
    z[:, :2] = torch.linspace(-1, 1, 10).view(1, -1).repeat(10, 1).t().reshape(-1, 2)
    z = z.view(num_samples, z_dim, 1, 1).to(gen.device)
    
    # Forward the samples through the generator
    with torch.no_grad():
        samples = gen.forward_given_samples(z)
    
    # Rescale samples to [0, 1] range
    samples = (samples / 2 + 0.5)
    
    # Save out an image holding all 100 samples.
    torchvision.utils.save_image.save_image(samples, path, nrow=10, padding=2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
