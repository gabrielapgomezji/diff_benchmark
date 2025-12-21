from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def create_trainer(
    max_epochs: int = 50,
    monitor: str ="val_accuracy",
    mode: str ="max",
    patience: int =5,
    accelerator: str ="auto",
    devices: str ="auto",
    save_dir: str ="./data/results/checkpoints",
) -> Trainer:
    """Creates a PyTorch Lightning Trainer with early stopping and model checkpointing."""
    early_stop = EarlyStopping(
        monitor=monitor, mode=mode, patience=patience, verbose=True
    )
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=1,
        monitor=monitor,
        mode=mode,
        filename="best_model_path",
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop, checkpoint],
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
        fast_dev_run=False,
        val_check_interval=0.5,
    )
    return trainer
