import torch
from models.vae import VAE
from utils.dataparser import load_yaml, load_data, parse_data
from engine.trainer import train
import argparse
from pathlib import Path



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    epochs = int(train_cfg.get("epchs",5))
    batch_size = int(train_cfg.get("batch_size", 100))
    input_dims = int(train_cfg.get("input_dim", 1))
    hidden_dims = int(train_cfg.get("hidden_dim",1))
    latent_dims = int(train_cfg.get("latent_dim",1))
    train_size = float(train_cfg.get("train_size", 0.8))

    path = Path(data_cfg.get("path", "/SF2935project/data"))
    file_name = str(data_cfg.get("file", "frey_rawface.mat"))

    data_path = path / file_name

    data = load_data(data_path)
    train_loader, test_loader = parse_data(train_size, data, batch_size)

    model = VAE(input_dims, hidden_dims, latent_dims)

    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9)

    loss_list = train(train_loader, epochs, model, optimizer, batch_size)

    print(len(loss_list))



if __name__ == "__main__":
    main()