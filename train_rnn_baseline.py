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
from models_cartpole import build_model, RNNModel
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
from trainSequential_ebonye_cartpole_zerodyn import count_files_in_folder, load_dataset_full, load_dataset_chunk, SegmentedCartpoleDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.backends.cudnn.benchmark = True

def train(model, args):
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.training.learning_rate,
                                  weight_decay=1e-4)
    curriculum = Curriculum(args.training.curriculum)
    loss_func = getattr(tasks, args.loss, None)
    assert loss_func is not None, f"Loss function {args.loss} not found in tasks module"

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
                        loss, _, grad_norm, prev_grad_norm = train_step(model, xs, ys, optimizer, loss_func, current_step, args, num_training_steps) 
                        
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
                        
                        elif current_step % 25000 == 0 and not args.test_run and current_step >= 5000 and local_rank == 0:
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

def train_step(model, xs, ys, optimizer, state_loss, current_step, args, num_training_steps):
    optimizer.zero_grad()
    
    # Normalizing Data
    state_max_scale = [7.0, 8.0, 1.0, 1.0, 5.0]
    control_max_scale = 15.0
    xs_sc = xs / torch.tensor(state_max_scale, device=xs.device)
    ys_sc = ys / torch.tensor([control_max_scale, 1.0], device=ys.device)
    ys_sc_for_model = ys_sc.clone()
    ys_sc_for_model[..., 1] = ys_sc_for_model[..., 1] + 1 # -1, 0, 1 -> 0, 1, 2

    # Forward Pass
    s, m, a = xs_sc, ys_sc_for_model[..., 1].unsqueeze(-1), ys_sc_for_model[..., 0].unsqueeze(-1)
    s_pred, m_logits, a_pred = model(s, m, a)
    # s_pred, m_logits, a_pred = s_pred.detach(), m_logits.detach(), a_pred.detach()

    # Mask for Zero-Dynamics Indices (assuming label at idx 1 and -1 means zero-dynamics)
    zero_dyn_mask = ys_sc[..., 1] == -1  
    zero_dyn_mask = zero_dyn_mask.unsqueeze(-1)

    # Control Input Regression MSE Loss (mask is used so that predicted actions during zero-dynamics timesteps are not penalized)
    ys_sc = ys_sc.to(a_pred.device)
    raw_mse = (a_pred.squeeze(-1)[:,:-1] - ys_sc[:, :-1, 0]).pow(2)
    active_mask = (~zero_dyn_mask[:, :-1].squeeze(-1)).float().to(raw_mse.device)  
    loss_controls = (raw_mse * active_mask).sum() / (active_mask.sum()+1e-8)

    # State Regression Loss
    xs_sc = xs_sc.to(s_pred.device)
    loss_states = state_loss(s_pred[:,:-1], xs_sc[:,1:])

    # Mode Classification Cross-Entropy Loss
    target_modes = (ys_sc[:, :-1, 1] + 1).long()
    mode_logits_flat = m_logits[:, :-1, :].reshape(-1, 3)
    target_modes_flat = target_modes.reshape(-1)
    loss_switch = nn.CrossEntropyLoss()(mode_logits_flat, target_modes_flat)

    # Total Loss
    alpha_controls = 1.0
    alpha_states = 5.0
    alpha_switch = 1.0
    loss = alpha_controls * loss_controls + alpha_states * loss_states + alpha_switch * loss_switch

    # Backward Pass
    loss.backward()
    prev_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    optimizer.step()

    return loss.detach().item(), (s_pred.detach(), m_logits.detach(), a_pred.detach()), grad_norm, prev_grad_norm

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
        end_index += remainder
    
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

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 10
    
    model = RNNModel(args.model.ndims, args.model.hidden_size, args.model.n_embd)

    dist.init_process_group(backend='nccl')
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if local_rank == 0 and not args.test_run:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
            id=run_id if run_id is not None else None
        )

    dist.barrier() 

    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    model.train()
    train(model, args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "rnn"]
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

        train_source_path = "train_rnn_baseline.py"
        train_dest_path = os.path.join(args.out_dir, "train_rnn_baseline.py")
        shutil.copy(train_source_path, train_dest_path)

    main(args)