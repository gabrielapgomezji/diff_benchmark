from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def create_trainer(
    max_epochs=50,
    monitor="val_accuracy",
    mode="max",
    patience=5,
    accelerator="auto",
    devices="auto",
    save_dir="./data/results/checkpoints",
):
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
