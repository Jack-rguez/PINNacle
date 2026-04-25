import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde
from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.chaotic import KuramotoSivashinskyEquation
from src.pde.heat import Heat2D_ComplexGeometry
from src.pde.ns import NS2D_LidDriven
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from src.utils.rar import rar_wrapper

# HPIT benchmark PDE set: exactly the 5 PDEs used in the neural operator comparison.
# Do NOT modify benchmark.py — this file is the HPIT-specific entry point.
# Created per CLAUDE_CODE_PRERUN_FIXES.md Fix 7 (2026-04-18).
pde_list = [
    Burgers1D,
    Burgers2D,
    Heat2D_ComplexGeometry,
    KuramotoSivashinskyEquation,
    NS2D_LidDriven,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNBench trainer — HPIT 5-PDE set')
    parser.add_argument('--name', type=str, default="benchmark")
    parser.add_argument('--device', type=str, default="0")  # set to "cpu" enables cpu training
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default="100*5")
    parser.add_argument('--loss-weight', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iter', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--method', type=str, default="adam")

    command_args = parser.parse_args()

    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

    for pde_config in pde_list:

        def get_model_dde():
            if isinstance(pde_config, tuple):
                pde = pde_config[0](**pde_config[1])
            else:
                pde = pde_config()

            if command_args.method == "gepinn":
                pde.use_gepinn()

            net = dde.nn.FNN([pde.input_dim] + parse_hidden_layers(command_args) + [pde.output_dim], "tanh", "Glorot normal")
            if command_args.method == "laaf":
                net = DNN_LAAF(len(parse_hidden_layers(command_args)) - 1, parse_hidden_layers(command_args)[0], pde.input_dim, pde.output_dim)
            elif command_args.method == "gaaf":
                net = DNN_GAAF(len(parse_hidden_layers(command_args)) - 1, parse_hidden_layers(command_args)[0], pde.input_dim, pde.output_dim)
            net = net.float()

            loss_weights = parse_loss_weight(command_args)
            if loss_weights is None:
                loss_weights = np.ones(pde.num_loss)
            else:
                loss_weights = np.array(loss_weights)

            opt = torch.optim.Adam(net.parameters(), command_args.lr)
            if command_args.method == "multiadam":
                opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])
            elif command_args.method == "lra":
                opt = LR_Adaptor(opt, loss_weights, pde.num_pde)
            elif command_args.method == "ntk":
                opt = LR_Adaptor_NTK(opt, loss_weights, pde)
            elif command_args.method == "lbfgs":
                opt = Adam_LBFGS(net.parameters(), switch_epoch=5000, adam_param={'lr':command_args.lr})

            model = pde.create_model(net)
            model.compile(opt, loss_weights=loss_weights)
            if command_args.method == "rar":
                model.train = rar_wrapper(pde, model, {"interval": 1000, "count": 1})
            return model

        trainer.add_task(
            get_model_dde, {
                "iterations": command_args.iter,
                "display_every": command_args.log_every,
                "callbacks": [
                    TesterCallback(log_every=command_args.log_every),
                    PlotCallback(log_every=command_args.plot_every, fast=True),
                    LossCallback(verbose=True),
                ]
            }
        )

    trainer.setup(__file__, seed)
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary()
