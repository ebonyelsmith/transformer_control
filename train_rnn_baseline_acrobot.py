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
from models_acrobot_new import build_model, RNNModel
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
from trainSequential_ebonye_acrobot import count_files_in_folder, load_dataset_chunk, load_dataset_full, SegmentedAcrobotDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    num_chunks = total_files 
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
                # dataset_full {(states, (control input, mode), link1_length, link2_length, link1_mass, link2_mass)}
                dataset, dataset_full, link_length1, link_length2, link_mass1, link_mass2 = load_chunk(chunk_idx, num_chunks, files_per_chunk, remainder, fullpicklepath)

                # Batch from chunk
                # xs_tensor : (t1, t2, t1., t2.)     [B, 120, 4]
                # ys_tensor : (u, m)                 [B, 120, 2]
                # link_lengths1_tensor : (l1)        [B]
                # link_lengths2_tensor : (l2)        [B]
                # link_masses1_tensor : (m1)         [B]
                # link_masses2_tensor : (m2)         [B]
                xs_tensor, ys_tensor, link_lengths1, link_lengths2, link_masses1, link_masses2 = dataset_full[0]
                link_lengths1_tensor = torch.tensor(np.array(link_lengths1), device=xs_tensor.device).squeeze(0)
                link_lengths2_tensor = torch.tensor(np.array(link_lengths2), device=xs_tensor.device).squeeze(0)
                link_masses1_tensor = torch.tensor(np.array(link_masses1), device=xs_tensor.device).squeeze(0)
                link_masses2_tensor = torch.tensor(np.array(link_masses2), device=xs_tensor.device).squeeze(0)

                # Preprocessing
                # xs_tensor : (cos t1, sin t1, cos t2, sin t2, t1., t2.)   [B, 120, 6]  
                xs_tensor, ys_tensor, link_lengths1_tensor, link_lengths2_tensor, link_masses1_tensor, link_masses2_tensor = preprocess(xs_tensor, ys_tensor, link_lengths1_tensor, link_lengths2_tensor, link_masses1_tensor, link_masses2_tensor)
                dataset_full = TensorDataset(xs_tensor, ys_tensor, link_lengths1_tensor, link_lengths2_tensor, link_masses1_tensor, link_masses2_tensor)


                # Trajectories are processed, but not normalized.
                sampler = DistributedSampler(dataset_full, shuffle=True)
                dataloader = DataLoader(dataset_full, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
                with tqdm(total=len(dataloader), desc=f"Training Chunk {chunk_idx + 1}/{num_chunks}") as pbar:
                    for xs, ys, _, _, _, _ in dataloader:
                        # xs [b, 120, 5], ys [b, 120, 2]
                        loss, loss_a, loss_s, loss_m, _, grad_norm, prev_grad_norm = train_step(model, xs, ys, optimizer, loss_func, current_step, args, num_training_steps) 
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
                                    "train_loss": loss,
                                    "train_control_mse": loss_a,
                                    "train_state_mse": loss_s,
                                    "train_mode_ce": loss_m,
                                    "grad_norm": grad_norm,
                                }
                            )
                        
                        if current_step % args.wandb.log_val_every_steps == 0 and not args.test_run and local_rank == 0:
                            val_loss, val_loss_s, val_loss_a, val_loss_m = validate(model, args)
                            wandb.log(
                                {
                                    "step": current_step,
                                    "train_loss": loss,
                                    "train_control_mse": loss_a,
                                    "train_state_mse": loss_s,
                                    "train_mode_ce": loss_m,
                                    "val_loss": val_loss,
                                    "val_loss_s": val_loss_s,
                                    "val_loss_a": val_loss_a,
                                    "val_loss_m": val_loss_m,
                                    "grad_norm": grad_norm,
                                }
                            )

                        if current_step % args.training.save_every_steps == 0 and not args.test_run and current_step <= 5000 and local_rank == 0:
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
    # s1, a1, m1 -> s1, a1, m1, s2
    # xs_tensor : (cos t1, sin t1, cos t2, sin t2, t1., t2.)   [B, 120, 6]  
    # ys_tensor : (u, m)                 [B, 120, 2]
    optimizer.zero_grad()
    xs, ys = xs.cuda(), ys.cuda()

    # Normalizing Data
    state_max_scale = [1.0, 1.0, 1.0, 1.0, 7.0, 7.0]
    control_max_scale = 10.0
    xs_sc = xs / torch.tensor(state_max_scale, device=xs.device)
    ys_sc = ys / torch.tensor([control_max_scale, 1.0], device=ys.device)
    ys_sc_for_model = ys_sc.clone()
    ys_sc_for_model[..., 1] = ys_sc_for_model[..., 1] + 1 # -1, 0, 1 -> 0, 1, 2

    # Forward Pass
    s, m, a = xs_sc, ys_sc_for_model[:, :, 1].unsqueeze(-1), ys_sc_for_model[:, :, 0].unsqueeze(-1)
    # (s1, ..., s120), (m1, ..., m120), (a1, ..., a120)
    s_pred, m_logits, a_pred = model(s, m, a)
    # (spred2, ..., spred121), (mpred1, ..., mpred120), (apred1, ..., apred120)
    s_pred = s_pred[:, :-1, :]
    # (s_pred2, ..., spred120)

    # Mask for Zero-Dynamics Indices (assuming label at idx 1 and -1 means zero-dynamics)
    zero_dyn_mask = ys_sc[..., 1] == -1  
    zero_dyn_mask = zero_dyn_mask.unsqueeze(-1)

    # Control Input Regression MSE Loss (mask is used so that predicted actions during zero-dynamics timesteps are not penalized)
    ys_sc = ys_sc.to(a_pred.device)
    raw_mse = (a_pred.squeeze(-1)[:, :-1] - ys_sc[:, :-1, 0]).pow(2) # (a_pred1 - a1)**2 ... (a_pred119 - a119)**2
    active_mask = (~zero_dyn_mask[:, :-1].squeeze(-1)).float().to(raw_mse.device)  
    loss_controls = (raw_mse * active_mask).sum() / (active_mask.sum()+1e-8)

    # State Regression Loss
    xs_sc = xs_sc.to(s_pred.device)
    loss_states = state_loss(s_pred, xs_sc[:,1:]) # (s_pred2 - s2)**2 + ... + (s_pred120 - s120)**2

    # Mode Classification Cross-Entropy Loss
    target_modes = (ys_sc[:, :-1, 1] + 1).long() # m1, ..., m119
    mode_logits_flat = m_logits[:, :-1, :].reshape(-1, 3) # mpred1, ..., mpred119
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

    return loss.detach().item(), loss_controls.detach(), loss_states.detach().item(), loss_switch.detach().item(), (s_pred.detach(), m_logits.detach(), a_pred.detach()), grad_norm, prev_grad_norm

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
               link_lengths1_tensor : torch.tensor,
               link_lengths2_tensor : torch.tensor,
               link_masses1_tensor : torch.tensor,
               link_masses2_tensor):
    
    # xs : (t1, t2, t1., t2.)     [B, 120, 4] -> (cos t1, sin t1, cos t2, sin t2, t1., t2.)
    # ys : (u, m)                 [B, 120, 2] ->
    # link_lengths1_tensor : (l1) [B]
    # link_lengths2_tensor : (l2) [B]
    # link_masses1_tensor : (m1)  [B]
    # link_masses2_tensor : (m2)  [B]
    
    # Compute sin/cos Components
    xs_cos_theta1_tensor = torch.cos(xs[:, :, 0])
    xs_sin_theta1_tensor = torch.sin(xs[:, :, 0])
    xs_cos_theta2_tensor = torch.cos(xs[:, :, 1])
    xs_sin_theta2_tensor = torch.sin(xs[:, :, 1])
    xs_tensor_updated = torch.cat((xs_cos_theta1_tensor.unsqueeze(-1), xs_sin_theta1_tensor.unsqueeze(-1), xs_cos_theta2_tensor.unsqueeze(-1), xs_sin_theta2_tensor.unsqueeze(-1), xs[:, :, 2:]), dim=-1)

    return xs_tensor_updated, ys, link_lengths1_tensor, link_lengths2_tensor, link_masses1_tensor, link_masses2_tensor


def validate(model, args):
    model.eval()

    id_loss = 0.0
    id_state_loss = 0.0
    id_control_loss = 0.0
    id_mode_loss = 0.0
    total_samples_id = 0

    state_loss = getattr(tasks, args.loss, None)
    id_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder_test)
    pickle_file = os.path.join(id_data_dir, 'batch_test_0_25.pkl')
    with open(pickle_file, 'rb') as file:
        id_data = pickle.load(file)

    id_data = TensorDataset(id_data[0], id_data[1], torch.tensor(id_data[2]), torch.tensor(id_data[3]), torch.tensor(id_data[4]), torch.tensor(id_data[5]))
    id_loader = DataLoader(id_data, batch_size=64, shuffle=False)
    for xs, ys, _, _, _, _ in id_loader:
        with torch.no_grad():
            xs, ys = xs[:, :120, :], ys[:, :120, :]
            
            xs_cos_theta1_tensor = torch.cos(xs[:, :, 0])
            xs_sin_theta1_tensor = torch.sin(xs[:, :, 0])
            xs_cos_theta2_tensor = torch.cos(xs[:, :, 1])
            xs_sin_theta2_tensor = torch.sin(xs[:, :, 1])
            xs = torch.cat((xs_cos_theta1_tensor.unsqueeze(-1), xs_sin_theta1_tensor.unsqueeze(-1), xs_cos_theta2_tensor.unsqueeze(-1), xs_sin_theta2_tensor.unsqueeze(-1), xs[:, :, 2:]), dim=-1)

            xs, ys = xs.cuda(), ys.cuda()

            # Normalizing Data
            state_max_scale = [1.0, 1.0, 1.0, 1.0, 7.0, 7.0]
            control_max_scale = 10.0
            xs_sc = xs / torch.tensor(state_max_scale, device=xs.device)
            ys_sc = ys / torch.tensor([control_max_scale, 1.0], device=ys.device)
            ys_sc_for_model = ys_sc.clone()
            ys_sc_for_model[..., 1] = ys_sc_for_model[..., 1] + 1 # -1, 0, 1 -> 0, 1, 2

            # Forward Pass
            s, m, a = xs_sc, ys_sc_for_model[:, :, 1].unsqueeze(-1), ys_sc_for_model[:, :, 0].unsqueeze(-1)
            # (s1, ..., s120), (m1, ..., m120), (a1, ..., a120)
            s_pred, m_logits, a_pred = model(s, m, a)
            # (spred2, ..., spred121), (mpred1, ..., mpred120), (apred1, ..., apred120)
            s_pred = s_pred[:, :-1, :]
            # (s_pred2, ..., spred120)

            # Mask for Zero-Dynamics Indices (assuming label at idx 1 and -1 means zero-dynamics)
            zero_dyn_mask = ys_sc[..., 1] == -1  
            zero_dyn_mask = zero_dyn_mask.unsqueeze(-1)

            # Control Input Regression MSE Loss (mask is used so that predicted actions during zero-dynamics timesteps are not penalized)
            ys_sc = ys_sc.to(a_pred.device)
            raw_mse = (a_pred.squeeze(-1)[:, :-1] - ys_sc[:, :-1, 0]).pow(2) # (a_pred1 - a1)**2 ... (a_pred119 - a119)**2
            active_mask = (~zero_dyn_mask[:, :-1].squeeze(-1)).float().to(raw_mse.device)  
            loss_controls = (raw_mse * active_mask).sum() / (active_mask.sum()+1e-8)

            # State Regression Loss
            xs_sc = xs_sc.to(s_pred.device)
            loss_states = state_loss(s_pred, xs_sc[:,1:]) # (s_pred2 - s2)**2 + ... + (s_pred120 - s120)**2

            # Mode Classification Cross-Entropy Loss
            target_modes = (ys_sc[:, :-1, 1] + 1).long() # m1, ..., m119
            mode_logits_flat = m_logits[:, :-1, :].reshape(-1, 3) # mpred1, ..., mpred119
            target_modes_flat = target_modes.reshape(-1)
            loss_switch = nn.CrossEntropyLoss()(mode_logits_flat, target_modes_flat) 

            # Total Loss
            alpha_controls = 1.0
            alpha_states = 5.0
            alpha_switch = 1.0
            loss = alpha_controls * loss_controls + alpha_states * loss_states + alpha_switch * loss_switch

            batch_size = xs.size(0)           
            id_loss += (loss.item() * batch_size)
            id_state_loss += (loss_states.item() * batch_size)
            id_control_loss += (loss_controls.item() * batch_size)
            id_mode_loss += (loss_switch.item() * batch_size)
            total_samples_id += batch_size
    
    id_loss /= total_samples_id
    id_state_loss /= total_samples_id
    id_control_loss /= total_samples_id
    id_mode_loss /= total_samples_id

    del id_data
    torch.cuda.empty_cache()
    gc.collect()

    model.train()
    return id_loss, id_state_loss, id_control_loss, id_mode_loss


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 10
    
    model = RNNModel(args.model.ndims, 
                     args.model.hidden_size, 
                     args.model.num_layers, 
                     args.model.cell_type,
                     args.model.n_embd)

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

        model_source_path = "models_acrobot_new.py"
        model_dest_path = os.path.join(args.out_dir, "models_acrobot_new.py")
        shutil.copy(model_source_path, model_dest_path)

        train_source_path = "train_rnn_baseline_acrobot.py"
        train_dest_path = os.path.join(args.out_dir, "train_rnn_baseline.py")
        shutil.copy(train_source_path, train_dest_path)

    main(args)