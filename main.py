import argparse
import logging
import os
os.chdir('/home/yan/workspace/HiDrug')
from datetime import datetime

import wandb
import lightning as L
# from lightning import seed_everything
# from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# from model.hierarchy_ablation_rnn import HiDrug
from model.cs_hierarchy import HiDrug
# from model.hierarchy import HiDrug
from model.hierarchy_ablation import HiDrug_noHi
from model.hierarchy_ablation_text import HiDrug_noT
from data.dataset import build_dataloader
from model.utils import print_args, set_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument(
        "--dataset_name",
        default="mimic3",
        choices=["mimic3", "mimic4"],
        type=str,
    )
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--n_workers", default=4, type=int)
    # parser.add_argument("--topk_smiles", default=1, type=int)
    # model
    parser.add_argument("--model_name", default='HiDrug', type=str)
    # hyperparameter
    parser.add_argument("--graph_hidden_size", default=64, type=int)
    parser.add_argument("--text_hidden_size", default=768, type=int)
    parser.add_argument("--crossmodal_size", default=128, type=int)
    parser.add_argument("--rnn_layer", default=2, type=int)
    parser.add_argument("--lr", default=0.003, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--w_pos", default=0.1, type=float)
    parser.add_argument("--w_neg", default=0.5, type=float)
    parser.add_argument("--w_reg", default=0.01, type=float)
    parser.add_argument(
        "--epsilon",
        default=0.2,
        type=float,
        help="epsilon controls the strength of perturbation",
    )
    # exp
    parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument(
    #     "--interv_style",
    #     type=str,
    #     choices=["attention", "interpolation", "concat", "add"],
    #     default="concat",
    # )
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--version", type=str, default='test')
    parser.add_argument("--dev", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()
    return args


def buil_model(args, dataset):
    if args.model_name == 'HiDrug':
        return HiDrug(args, dataset, args.graph_hidden_size, args.rnn_layer, args.crossmodal_size, dropout=0.2, bidirectional=False)
    elif args.model_name == 'HiDrug_noHi':
        return HiDrug_noHi(args, dataset, args.graph_hidden_size, args.rnn_layer, args.crossmodal_size, dropout=0.2, bidirectional=False)
    elif args.model_name == 'HiDrug_noT':
        return HiDrug_noT(args, dataset, args.graph_hidden_size, args.rnn_layer, args.crossmodal_size, dropout=0.2, bidirectional=False)

def experiment(args):
    #log_dir = f"run_logs/{args.model_name}/ver-{args.version}/{args.ckpt_path}"
    #log_dir = f"run_logs/{args.model_name}/ver-{args.version}/{args.ckpt_path}"
    log_dir = f"run_logs/{args.model_name}/ver-{args.version}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    set_logger(log_dir)
    wandb_logger = WandbLogger(
        project="HiTDrug",
        config=args,
        group=f"{args.model_name}",
        job_type=f"{args.version}",
        name=f"{args.model_name}_drug_decoding(real)", #include_drug_hist(real)
        mode="online" if bool(args.wandb) else "disabled",
    )
    print_args(args)
    seed = L.seed_everything(args.seed)
    logger.info(f"Current PID: {os.getpid()}")
    logger.info(f"Global seed set to: {seed}")
    logger.info(f"CWD:{os.getcwd()}")
    dataset, train_loader, val_loader, test_loader = build_dataloader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        dev=bool(args.dev),
        seed=seed,
    )
    logger.info("")
    logger.info(dataset.stat())

    model = buil_model(args, dataset)

    
    callbacks = []
    ckp_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoint",
        monitor=f"val/ja",
        mode="max",
    )
    callbacks.append(ckp_callback)

    trainer = L.Trainer(
        default_root_dir=log_dir,
        callbacks=callbacks,
        devices=[args.device],
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader) #ckpt_path=f"{log_dir}/checkpoint/epoch=49-step=300.ckpt"
    trainer.test(model, dataloaders=test_loader)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
    # original: f"run_logs/{args.model_name}/ver-{args.version}/2025-05-01_14-18-39/2025-05-02_00-59-41/
    # notext: f"run_logs/{args.model_name}/ver-{args.version/2025-05-02_00-59-41"epoch=29-step=180.ckpt