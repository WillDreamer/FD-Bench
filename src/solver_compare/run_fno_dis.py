import os
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from fdbench.dis_utils.fno_utils import *
from fdbench.models.neuralop.models import FNO

def train(args, model, train_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.end_learning_rate
    )

    loss_list = []
    runtime_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fnc = nn.MSELoss()

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch in train_dataloader:
            batch = batch.to(device)

            input = batch.x.float()
            input = input.permute(0, 3, 1, 2)
            target = batch.y.float()
            target = target.permute(0, 3, 1, 2)

            optimizer.zero_grad()
            out = model(input)

            loss = loss_fnc(out, target)
            # print(f"loss: {loss}")

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_runtime = time.time() - start_time

        avg_loss = epoch_loss / len(train_dataloader)
        loss_list.append(avg_loss)
        runtime_list.append(epoch_runtime)

        scheduler.step()

        if (epoch) % args.save_freq == 0:
            model_save_path = os.path.join(args.exp_name, f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch} to {model_save_path}")
    
        print(f"Epoch {epoch}/{args.num_epochs} - Loss: {avg_loss}, Runtime: {epoch_runtime:.2f}s")

    results = {"loss": loss_list, "runtime": runtime_list}
    txt_path = f"{args.exp_name}/results.txt"
    pkl_path = f"{args.exp_name}/results.pkl"

    with open(txt_path, "w") as f:
        for epoch, (loss, runtime) in enumerate(zip(loss_list, runtime_list)):
            f.write(f"Epoch {epoch+1}: Loss = {loss}, Runtime = {runtime:.2f}s\n")

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Training completed. Results saved to {txt_path} and {pkl_path}")



def main():
    parser = argparse.ArgumentParser(description="Argument parser for specifying dataset, model, and training configurations")

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--data_save_path", type=str)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--split_interval", type=int)
    parser.add_argument("--max_train_data", type=int)
    parser.add_argument("--stats_path", type=str)

    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--end_learning_rate", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--to_train", action="store_true")
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--input_channels", type=int)
    parser.add_argument("--output_channels", type=int)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--width", type=int)

    parser.add_argument("--use_huggingface", action="store_true")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
        print(f"Folder '{args.exp_name}' created.")
    else:
        print(f"Folder '{args.exp_name}' already exists.")

    model = FNO(n_modes=(args.modes, args.modes), hidden_channels=args.width,
                in_channels=args.input_channels, out_channels=args.output_channels)

    if args.to_train:
        train_dataloader = load_train(args)
        train(args, model, train_dataloader)

if __name__ == "__main__":
    main()

