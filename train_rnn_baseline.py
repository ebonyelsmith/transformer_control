# Imports from Ebonye file
import os
from random import randint
import uuid
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import tasks
from curriculum import Curriculum
from schema import schema
from models_cartpole import build_model
import wandb
import pickle
import random
import numpy as np
import torch
import gc
import json
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler
import shutil
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# My Imports
from trainSequential_ebonye_cartpole_zerodyn import count_files_in_folder, load_dataset_full, load_dataset_chunk, SegmentedCartpoleDataset, main

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True


def train(model, args):
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.training.learning_rate,
                                  weight_decay=1e-4)
    curriculum = Curriculum(args.training.curriculum)
    loss = getattr(tasks, args.loss, None)

    # Paths
    state_path = os.path.join(args.out_dir, "state.pt")
    dataset_folder = args.dataset_filesfolder
    picklefolder = args.pickle_folder
    fullpicklepath = os.path.join(dataset_folder, picklefolder)

    # Hyperparameters
    num_epochs = args.training.epochs
    batch_size_global = 64 

    # Batching
    batched_training_size = args.training.batch_size
    total_files = count_files_in_folder(fullpicklepath, "batch_", ".pkl")
    num_chunks = args.use_chunk  
    files_per_chunk = total_files // num_chunks
    remainder = total_files % num_chunks

    world_size = dist.get_world_size()
    batch_size = batch_size_global // world_size
    num_training_steps = num_epochs * total_files * batched_training_size // batch_size_global

    # Scheduler
    warmup_steps = int(0.01 * num_training_steps)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training Loop
    start_epoch = 0
    current_step = 0
    for epoch in range(start_epoch, num_epochs):
        chunk_to_resume = 0
        with tqdm(total=num_chunks - chunk_to_resume, desc="Chunk Progress") as chunk_pbar:
            for chunk_idx in range(chunk_to_resume, num_chunks):
                # dataset_full {(states, (control input, mode), cart_mass, pole_masses, pole_length)}
                dataset, dataset_full, cart_masses, pole_masses, pole_lengths = load_chunk(chunk_idx, num_chunks, files_per_chunk, remainder, fullpicklepath)

                # I don't really understand why this grabs a batch in the chunk.
                xs_tensor, ys_tensor, cartmasses, polemasses, polelengths = dataset_full[0]
                xs_tensor = xs_tensor.to("cpu")
                ys_tensor = ys_tensor.to("cpu")
                cartmasses_tensor = torch.tensor(np.array(cartmasses), device=xs_tensor.device)
                polemasses_tensor = torch.tensor(np.array(polemasses), device=xs_tensor.device)
                polelengths_tensor = torch.tensor(np.array(polelengths), device=xs_tensor.device)

                # Preprocessing
                xs_tensor, ys_tensor, cartmasses_tensor, polemasses_tensor, polelengths_tensor = preprocess(xs_tensor, ys_tensor, cartmasses_tensor, polemasses_tensor, polelengths_tensor)
                dataset_full = TensorDataset(xs_tensor, ys_tensor, cartmasses_tensor, polemasses_tensor, polelengths_tensor)
                segmented_dataset = SegmentedCartpoleDataset(dataset_full, window_size=120)

                # Trajectories are segmented and processed, but tuple elements not normalized.
                sampler = DistributedSampler(segmented_dataset, shuffle=True)
                dataloader = DataLoader(segmented_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
                with tqdm(total=len(dataloader), desc=f"Training Chunk {chunk_idx + 1}/{num_chunks}") as pbar:
                    for xs, ys, _, _, _ in dataloader:
                        loss, _, grad_norm, prev_grad_norm = train_step(model, xs, ys, optimizer, loss, current_step, args, num_training_steps) 
                        
                        lr_scheduler.step()
                        curriculum.update()
                        current_step += 1
                        pbar.update(1)

                        # Logging / Saving
                        local_rank = dist.get_rank()
                        if current_step % args.wandb.log_every_steps == 0 and not args.test_run and local_rank == 0:
                            wandb.log(
                                {
                                    "step": current_step,
                                    "loss": loss,
                                    "grad_norm": grad_norm,
                                }
                            )
                        
                        if current_step % args.training.save_every_steps == 0 and not args.test_run and current_step < 5000 and local_rank == 0:
                            training_state = {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "train_step": current_step,
                                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                                "epoch": epoch+1,
                                "loss": loss,
                            }
                            torch.save(training_state, state_path)
                            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}_step{current_step}.pt")
                            torch.save(training_state, checkpoint_path)
                            print(f"Checkpoint saved at epoch {epoch+1}, step {current_step}: {checkpoint_path}")
                        
                        elif current_step % 1000 == 0 and not args.test_run and current_step >= 5000 and local_rank == 0:
                            training_state = {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "train_step": current_step,
                                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                                "epoch": epoch+1,
                                "loss": loss,
                            }
                            torch.save(training_state, state_path)
                            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}_step{current_step}.pt")
                            torch.save(training_state, checkpoint_path)
                            print(f"Checkpoint saved at epoch {epoch+1}, step {current_step}: {checkpoint_path}")
                        
                # Cleanup
                print(f"Chunk {chunk_idx + 1}/{num_chunks} finished. unloading dataset from memory...")
                del dataset_full
                del dataset
                torch.cuda.empty_cache()
                gc.collect()
                chunk_pbar.update(1)

        print(f"============== Finished Epoch {epoch + 1}/{num_epochs} ==============\n")

    # Final Checkpoint
    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": current_step,
        "epoch": epoch+1,
        "loss": loss,
    }

    torch.save(training_state, state_path)
    checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}_step{current_step}.pt")
    torch.save(training_state, checkpoint_path)
    print(f"Final Checkpoint saved at epoch {epoch+1}, step {current_step}: {checkpoint_path}")              

def train_step(model, xs, ys, optimizer, loss, current_step, args, num_training_steps):
    ...

def load_chunk(chunk_index : int, 
               num_chunks : int, 
               files_per_chunk : int, 
               remainder : int, 
               pickle_path : str):
    if num_chunks == 1:
        return load_dataset_full(pickle_path)
    
    start_index = chunk_index * files_per_chunk
    end_index = (chunk_index + 1) * files_per_chunk - 1
    if chunk_index == num_chunks - 1:
        end_idx += remainder
    
    return load_dataset_chunk(pickle_path, start_index, end_index)


def preprocess(xs : torch.tensor, 
               ys : torch.tensor, 
               cart_masses : torch.tensor,
               pole_masses : torch.tensor,
               pole_lengths : torch.tensor):
    
    # Compute sin/cos Components
    xs_cos_tensor = torch.cos(xs[:, :, 2])
    xs_sin_tensor = torch.sin(xs[:, :, 2])
    xs_tensor_updated = torch.cat((xs[:, :, :2], xs_cos_tensor.unsqueeze(-1), xs_sin_tensor.unsqueeze(-1), xs[:, :, 3:]), dim=-1)

    # Throw out non-stabilizing sequences
    final_window = xs_tensor_updated[:, -40:, :5] 
    cos_theta_thresh = 0.9
    sin_theta_thresh = 0.5
    theta_dot_thresh = 1.0

    is_upright = (torch.abs(final_window[:, :, 2]) > cos_theta_thresh) & \
                    (torch.abs(final_window[:, :, 3]) < sin_theta_thresh) & \
                    (torch.abs(final_window[:, :, 4]) < theta_dot_thresh)
    
    mask = is_upright.all(dim=1) 

    return xs_tensor_updated[mask], ys[mask], cart_masses[mask], pole_masses[mask], pole_lengths[mask]

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run: 
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(args.out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

        model_source_path = "models_cartpole.py"
        model_dest_path = os.path.join(args.out_dir, "models_cartpole.py")
        shutil.copy(model_source_path, model_dest_path)

        train_source_path = "trainSequential_ebonye_cartpole_zerodyn.py"
        train_dest_path = os.path.join(args.out_dir, "trainSequential_ebonye_cartpole_zerodyn.py")
        shutil.copy(train_source_path, train_dest_path)

    main(args)