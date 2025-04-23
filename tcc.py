import click
import sys
import os
import yaml
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
import uuid

from examples.empenhos_df import EMPENHOS
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptdec.utils import cluster_accuracy
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import silhouette_score

@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=False
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=300,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=500,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode):
    writer = SummaryWriter()  # create the TensorBoard object
    # SummaryWriter is a class provided by PyTorch to log data 
    # callback function to call during training, uses writer from the scope
    
    # Open the configuration file and load the different arguments
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
    if testing_mode:
        num_clusters = config['num_clusters_testing']
    else:
        num_clusters = config['num_clusters']
    

    def training_callback(epoch, lr, loss, validation_loss): # summary writer
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    ds_train = EMPENHOS(
        train=True, testing_mode=testing_mode
    )  # training dataset
    ds_val = EMPENHOS(
        train=False, testing_mode=testing_mode
    )  # evaluation dataset

    
    """train_loader = DataLoader(ds_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=64)
    
    for i, batch in enumerate(train_loader):
        # batch is a tensor of shape (batch_size, 384)
        print(batch.shape)
        if i == 10:  # Stop after printing the first 3 batches
            break
    """

    autoencoder = StackedDenoisingAutoEncoder(
        [config['input_dim'], 1000, 2000, 2000, num_clusters], final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print("Pretraining stage.")
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9), #  it means the current update is made of 90% of the previous update (momentum) and 10% of the new gradient
        scheduler=lambda x: StepLR(x, 100, gamma=0.1), # gamma decay rate
        corruption=0.2, # introducing noise or modifying a percentage of the input data
    )
    print("Training stage.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train( #
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback, #
    )

    print("DEC stage.")
    model = DEC(cluster_number=num_clusters, hidden_dimension=10, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda,
    )
    
    #import pdb; pdb.set_trace()
    X, predicted = predict(
        ds_train, model, 1024, silent=True, return_actual=False, cuda=cuda
    )
    # np.bincount(predicted.cpu().numpy()).argmax() gives you the most frequent label
    
    
    
    # pensar em métodos de avaliação
    score = silhouette_score(X, predicted)
    print(f"Silhouette Score: {score:.4f}")
    
    
    
    # salvando o modelo para testes de inferência
    torch.save(autoencoder, os.path.join(config['saved_models_dir'], "autoencoder_full.pt"))
    torch.save(model, os.path.join(config['saved_models_dir'], "dec_model_full.pt"))
    
    writer.close()

if __name__ == "__main__":
    main()
