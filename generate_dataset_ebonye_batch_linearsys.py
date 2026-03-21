import os
import random
from tqdm import tqdm
from samplers import PendulumSampler
from samplers import LinearSystemSampler
from curriculum import Curriculum
from random import randint
import uuid
import ipdb

from quinine import QuinineArgumentParser
import torch
import yaml
from schema import schema
from models import build_model
import math
import random
import numpy as np
import torch
import pickle

# seed = [1]
torch.backends.cudnn.benchmark = True

def get_lambda1_and_lambda2_uniform(size=1, lambda1lowerbound=0.1, lambda1upperbound=1.0, lambda2lowerbound=1.0, lambda2upperbound=1.5): 
    """
    Samples valid masses and lengths for a pendulum system that meet specific constraints, using a uniform distribution.

    Args:
        size (int, optional): The number of samples to generate. Defaults to 1.
        lambda1lowerbound (float, optional): The lower bound for eigenvalue 1 of A matrix. Defaults to 0.1.
        lambda1upperbound (float, optional): The upper bound for eigenvalue 1 of A matrix. Defaults to 1.0.
        lambda2lowerbound (float, optional): The lower bound for eigenvalue 2 of A matrix. Defaults to 1.0001.
        lambda2upperbound (float, optional): The upper bound for eigenvalue 2 of A matrix. Defaults to 1.5.
    Returns:
        tuple: A tuple containing:
            - lambda1s (float): A valid eigenvalue 1 value sampled from the uniform distribution.
            - lambda2s (float): A valid eigenvalue 2 value sampled from the uniform distribution.
    """
    while True:
        
        lambda1s = sample_lambda_uniform(lambdalowerbound=lambda1lowerbound, lambdaupperbound=lambda1upperbound, size=size)
        lambda2s = sample_lambda_uniform(lambdalowerbound=lambda2lowerbound, lambdaupperbound=lambda2upperbound, size=size)
        
        return lambda1s, lambda2s



def sample_lambda_uniform(lambdalowerbound=0.1, lambdaupperbound=1.0, size=1):
    """
    Samples a valid eigenvalue for a pendulum system that meets specific constraints, using a uniform distribution.

    Args:
        lambdalowerbound (float, optional): The lower bound for the eigenvalue. Defaults to 0.1.
        lambdaupperbound (float, optional): The upper bound for the eigenvalue. Defaults to 1.0.
        size (int, optional): The number of samples to generate. Defaults to 1.

    Returns:
        float: A valid eigenvalue value sampled from the uniform distribution.
    """
    # Generate a random value between the specified bounds
    lambda_value = np.random.uniform(lambdalowerbound, lambdaupperbound, size=size)
    return lambda_value

def generate_random_X0(x1_range=(-1, 1), x2_range=(-1, 1)):
    """
    Generates a random initial state for a pendulum system.

    Returns:
        list: A list containing:
            - x1 (float): A randomly generated x1, sampled uniformly from range [-1, 1].
            - x2 (float): A randomly generated x2, sampled uniformly from the range [-1, 1].
    """
    x1 = np.random.uniform(*x1_range)
    x2 = np.random.uniform(*x2_range)
    return [x1, x2]


def save_pickle(data, pickle_path):
    """
    Saves data to a pickle file.

    Args:
        data (any): The data to save.
        pickle_path (str): The path to the pickle file where the data will be saved.
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


def append_to_dataset_logger(iteration, masses, lengths, xs_shape, log_file):
    """
    Appends simulation details to a seed file for tracking and reproducibility.

    Args:
        iteration (int): The iteration number of the simulation.
        masses (list[float]): The mass used in the
        lengths (list[float]): The length used in the simulation.
        # k_values (list[float]): The gain matrix used in the simulation.
        xs_shape (tuple): The shape of the state dataset (batch_size, n_dim, timepoints).
        log_file (str): The path to the log file.
    """
    with open(log_file, 'a') as f:
        f.write(f"Iteration: {iteration}\n")
        f.write("Masses: " + str(masses) + "\n")
        f.write("Lengths: " + str(lengths) + "\n")
        # f.write("K: " + str(k_values) + "\n")
        f.write(f"xs.shape: {xs_shape}\n")
        f.write("-" * 10 + "\n")


def make_train_data(args):
    """
    Geneates training datasets for an inverted pendulum system simulation and saves them 
    as pickle files along with metadata for reproducibility.

    Args:
        args (Namespace):
            - args.training.train_steps (int): The total number of pickle files.
            - args.training.batch_size (int): batch of each pickle file.
            - args.training.curriculum (dict): Curriculum settings to adjust training parameters dynamically.
            - args.dataset_filesfolder (str): The directory where dataset files and logs are stored.
            - args.dataset_logger_textfile (str): The name of the file for logging dataset info.
            - args.pickle_folder (str): The dataset_filesfolder subfolder where generated dataset pickle files are saved.

    Notes:
        - The `PendulumSampler` is used to generate the dataset based on random valid pendulum parameters (masses and lengths).
        - The generated datasets are saved as pickle files named in the format `multipendulum_{i}.pkl`.
        - Metadata such as the current seed, pendulum parameters, and dataset shape is logged in a separate file for reproducibility.
        - The curriculum dynamically updates training parameters, such as the number of points in each dataset.

    """
    curriculum = Curriculum(args.training.curriculum)
    starting_step = 0
    bsize = args.training.batch_size
    pbar = tqdm(range(starting_step, args.training.train_steps + args.training.test_pendulums + args.training.test_pendulums_outofdistr)) 
    # pbar_test = tqdm(range(args.training.test_pendulums))
    # num_test_pendulums = args.training.test_pendulums

    # seed_file = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    train_logger = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    test_logger = os.path.join(args.dataset_filesfolder, args.dataset_test_logger_textfile)
    # test_logger_outofdistr = os.path.join(args.dataset_filesfolder, args.dataset_test_outofdistr_logger_textfile) ## 3/5/2025 out of distribution data
    base_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder)
    test_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder_test)
    # test_data_dir_outofdistr = os.path.join(args.dataset_filesfolder, args.pickle_folder_test_outofdistr) ## 3/5/2025 out of distribution data
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    # os.makedirs(test_data_dir_outofdistr, exist_ok=True) ## 3/5/2025 out of distribution data

    b_size = args.training.batch_size
    for i in pbar:
        # reseed_all(seed[0]+i)
        
        # masses, lengths = get_valid_masses_and_lengths_uniform()
        # # sampler = PendulumSampler(n_dims=2, init_conditions=X0)
        # sampler = PendulumSampler(n_dims=2)
        # # T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, bsize, mass = masses, length = lengths)
        # T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
        
        if i < args.training.train_steps:
            
            lambda1s, lambda2s = get_lambda1_and_lambda2_uniform(size=b_size)
            ### constant system
            # lambda1s = np.zeros(b_size)
            # lambda2s = np.zeros(b_size)
            ###
            # sampler = PendulumSampler(n_dims=2)
            sampler = LinearSystemSampler(n_dims=2)
            T, xs, control_values = sampler.generate_xs_dataset(curriculum.n_points, b_size, lambda1=lambda1s, lambda2=lambda2s)
            pickle_file = f'batch_{i}.pkl'
            pickle_path = os.path.join(base_data_dir, pickle_file)
            # save_pickle((xs, control_values, masses, lengths), pickle_path)
            save_pickle((xs, control_values, lambda1s, lambda2s), pickle_path)
            # import ipdb; ipdb.set_trace()
            # append_to_dataset_logger(i, masses, lengths, xs.shape, train_logger)
            append_to_dataset_logger(i, lambda1s, lambda2s, xs.shape, train_logger)
        

        elif i >= args.training.train_steps and i < args.training.train_steps + args.training.test_pendulums:
            lambda1s, lambda2s = get_lambda1_and_lambda2_uniform(size=b_size)
            ### constant system
            # lambda1s = np.zeros(b_size)
            # lambda2s = np.zeros(b_size)
            ###
            # X0 = generate_random_X0(theta_range=(-np.pi, np.pi), thetadot_range=(-3, 3))
            # sampler = PendulumSampler(n_dims=2, init_conditions=X0)
            # sampler = PendulumSampler(n_dims=2)
            sampler = LinearSystemSampler(n_dims=2)
            T, xs, control_values = sampler.generate_xs_dataset(curriculum.n_points, bsize, lambda1=lambda1s, lambda2=lambda2s)
            # T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)

            # pickle_file = f'multipendulum_test_{i-args.training.train_steps}.pkl'
            pickle_file = f'batch_test_{i-args.training.train_steps}.pkl'
            pickle_path = os.path.join(test_data_dir, pickle_file)

            # save_pickle((xs, control_values, masses, lengths), pickle_path)
            save_pickle((xs, control_values, lambda1s, lambda2s), pickle_path)
            # append_to_dataset_logger(i-args.training.train_steps, masses, lengths, xs.shape, test_logger)
            append_to_dataset_logger(i-args.training.train_steps, lambda1s, lambda2s, xs.shape, test_logger)
        else:
            # masses, lengths = get_valid_masses_and_lengths_uniform(size=b_size, masslowerbound=2.07, massupperbound=3.07, lengthlowerbound=2.3, lengthupperbound=3.3) ## 3/5/2025 out of distribution data
            # sampler = PendulumSampler(n_dims=2)
            # # T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
            # T, xs, control_values = sampler.generate_xs_dataset(curriculum.n_points, bsize, mass = masses, length = lengths)
            # # pickle_file = f'multipendulum_test_outofdistr_{i-args.training.train_steps-args.training.test_pendulums}.pkl'
            # pickle_file = f'batch_test_outofdistr_{i-args.training.train_steps-args.training.test_pendulums}.pkl'
            # # xs_scaled = [torch.squeeze(xs)[i]/(5*0.98**i) for i in range(len(torch.squeeze(xs)))]
            # # xs_scaled = torch.unsqueeze(torch.stack(xs_scaled), 0)
            # # control_scaled = [torch.squeeze(control_values)[i]/(5*0.98**i) for i in range(len(torch.squeeze(control_values)))]
            # # control_scaled = torch.unsqueeze(torch.stack(control_scaled), 0)
            # pickle_path = os.path.join(test_data_dir_outofdistr, pickle_file)
            # # unscaled_pickle_path = os.path.join(unscaled_test_data_ood_dir, pickle_file)
            # # save_pickle((xs_scaled, control_scaled, masses, lengths), pickle_path)
            # # save_pickle((xs, control_values, masses, lengths), unscaled_pickle_path)
            # # save_pickle((xs, control_values), pickle_path)
            # save_pickle((xs, control_values, masses, lengths), pickle_path)
            # append_to_dataset_logger(i-args.training.train_steps-args.training.test_pendulums, masses, lengths, xs.shape, test_logger_outofdistr)
            pass 

        # append_to_seed_file(seed_file, i, seed[0] + i, masses, lengths, k_values, xs.shape)
        curriculum.update()
    
    

def main(args):
    # reseed_all(seed[0])
    global_seed = 42
    random.seed(global_seed)
    make_train_data(args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    os.makedirs(args.dataset_filesfolder, exist_ok=True)
    with open(os.path.join(args.dataset_filesfolder, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)

