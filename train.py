import torch
from torch import nn
from einops import rearrange

from opt import get_opts
from models import GCN, SAGE
# datasets
from dataset import GraphDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sampler import NeighborSampler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




class GCNSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams) # params -> self.parmas

        # self.gcn = GCN(
        #     input_dim=hparams.input_dim, 
        #     hidden_dim_ls=hparams.hidden_dim_ls, 
        #     output_dim=hparams.output_dim)

        
    def setup(self, stage=None):
        self.dataset = GraphDataset(
            hparams.graph_name, split="all", ratio=hparams.ratio)
        self.train_dataset = GraphDataset(
            hparams.graph_name, split="train", ratio=hparams.ratio)
        self.val_dataset = GraphDataset(
            hparams.graph_name, split="val", ratio=hparams.ratio)
        
        

        self.edge_index = self.train_dataset.edge_index

        self.hidden_layer_num = hparams.hidden_layer_num

        self.sample_neighbor_num = hparams.sample_neighbor_num
        self.train_idx = self.train_dataset.node_ids
        self.val_idx = self.val_dataset.node_ids
        
        self.sage_neighsampler_parameters = {
                                        'num_layers':hparams.hidden_layer_num,
                                        'hidden_channels':hparams.hidden_dim,
                                        'dropout':0.0,
                                        'batchnorm': False,
                                        }
        self.sage = SAGE(in_channels=self.dataset.d, out_channels=self.dataset.c, **self.sage_neighsampler_parameters)
    def forward(self, x, adjs): # TODO
        return self.sage(x, adjs)
        
    def train_dataloader(self):
        return NeighborSampler(self.edge_index, 
                               node_idx=self.train_idx, 
                               sizes=[self.sample_neighbor_num]*self.hidden_layer_num, 
                               batch_size=self.hparams.batch_size,
                               shuffle=True, 
                               num_workers=4
                               )
        # return DataLoader(self.train_dataset,
        #                   shuffle=True,
        #                   num_workers=4,
        #                   batch_size=self.hparams.batch_size,
        #                   pin_memory=True)

    def val_dataloader(self):
        return NeighborSampler(self.edge_index, 
                        node_idx=self.val_idx, 
                        sizes=[self.sample_neighbor_num]*self.hidden_layer_num, 
                        batch_size=self.hparams.batch_size,
                        shuffle=False, 
                        num_workers=4
                        )
        # return DataLoader(self.val_dataset,
        #                   shuffle=False,
        #                   num_workers=4,
        #                   batch_size=self.hparams.batch_size,
        #                   pin_memory=True)

    def configure_optimizers(self):
        self.opt = Adam(self.sage.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(self.opt, hparams.num_epochs, hparams.lr/1e2)

        return [self.opt], [scheduler]

    def training_step(self, batch, batch_idx): # TODO:batch_idx 第几个batch
        # batch 就是 getitem出来的东西，是个dict
        # 只负责写loss
        batch_size, n_id, adjs = batch
        y_true = self.dataset.y[n_id[:batch_size]]

        # self调用forward函数
        output = self(self.dataset.x[n_id], adjs)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, y_true)
        pred = output.argmax(dim=1)
        acc = sum(pred == y_true)/pred.shape[0]
        

        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size, n_id, adjs = batch
        y_true = self.dataset.y[n_id[:batch_size]]

        # self调用forward函数
        output = self(self.dataset.x[n_id], adjs)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, y_true)
        pred = output.argmax(dim=1)
        acc = sum(pred == y_true)/pred.shape[0]

        log = {'val_loss': loss,
               'val_acc': acc}
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        mean_acc = torch.tensor([x['val_acc'] for x in outputs]).mean()
    
        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/acc', mean_acc, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()
    system = GCNSystem(hparams)

    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=0,
                      log_every_n_steps=1,
                      check_val_every_n_epoch=20,
                      benchmark=True)

    trainer.fit(system)

    # test
    # ckpt_path = "./logs/exp/version_0/checkpoints/epoch=19-step=3140.ckpt"
    # # system = CoordMLPSystem(hparams)
    # system = CoordMLPSystem.load_from_checkpoint(ckpt_path)
    
    # # trainer = Trainer()
    # # 自动恢复模型,可以继续训练
    # # trainer.fit(model, ckpt_path=ckpt_path)
    # mlp = system.mlp
    # mlp.eval()
    # image = imageio.imread("/Users/sutongtong/Documents/GitHub/Coordinate-MLPs/images/fox.jpg")
    # resolution = 128
    # pred = torch.zeros([resolution, resolution, 3])
    # for i in range(resolution):
    #     for j in range(resolution):
    #         rgb = mlp(torch.FloatTensor([[i/resolution, j/resolution]]))
    #         pred[i, j, :] = rgb
    # print(pred.shape)
    # imageio.imwrite("./images/output.jpg", pred.detach().numpy())
   
    