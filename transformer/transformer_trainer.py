import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from datetime import datetime

from transformer.dataset import *
from transformer.transformer import (
	Transformer, create_pad_mask, create_look_ahead_mask
)


__SOS_IMAGE_TOKEN__  = 64
__EOS_IMAGE_TOKEN__  = 65
__MASK_IMAGE_TOKEN__ = 66

__SOS_TEXT_TOKEN__   = 67
__EOS_TEXT_TOKEN__   = 68
__PAD_TEXT_TOKEN__   = 69


def get_latest_snapshot_name(path):
    """
    A function to get the name of a latest snapshot file
    """

    if not os.path.isabs(path): path = os.path.join(os.getcwd(), path)
    snapshots = [os.path.join(path, s) for s in os.listdir(path)]

    if not snapshots: raise RuntimeError('No snapshots found')
    latest_snapshot = max(snapshots, key=os.path.getctime)
  
    return latest_snapshot


class Trainer():

    def __init__(
        self,
        model,
        optimizer,
        device, 
        train_dataset,
        val_dataset=None,
        gradient_clipping=None,
        snapshot_path=None   
    ):

        """ 
        :param model: a transformer model to train
        :type model : torch.nn.Module

        :type train_dataset: Text2ImageDataset
        :type val_dataset  : Text2ImageDataset

        """

        default_optimizer_params = {'lr': 1e-4}

        self.model  = model
        self.device = device

        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
   
        self.optimizer = optimizer
        
        self.gradient_clipping = gradient_clipping
        self.snapshot_path     = snapshot_path
        # internal snapshot parameters
        self.date_format       = '%Y-%m-%d_%H-%M-%S'


    def load_latest_snapshot(self):

        sname    = get_latest_snapshot_name(self.snapshot_path)
        snapshot = torch.load(sname)

        error_msg_header = f'Error loading snapshot {sname}' +\
                            '- incompatible snapshot format. '
        if 'optimizer' not in snapshot:
            raise KeyError(error_msg_header + 'Key "optimizer" is missing')
        if 'model' not in snapshot:
            raise KeyError(error_msg_header + 'Key "model" is missing')

        self.model.load_state_dict(snapshot['model'])
        self.optimizer.load_state_dict(snapshot['optimizer'])


    def save_model(self, replace_latest=False):

        if self.snapshot_path is None: return
                    
        time_string = datetime.now().strftime(self.date_format)

        states = {
            'model'    : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        if not replace_latest:
            torch.save(states, os.path.join(self.snapshot_path, time_string + '.pth'))
        else:
            try:
                os.remove(get_latest_snapshot_name(self.snapshot_path))
            except Exception:
                pass
            torch.save(states, os.path.join(self.snapshot_path, time_string + '.pth'))


    def train(self, n_epochs=100, batch_size=32, save_interval=1000, from_zero=True, plot_loss_history=True):

        MAX_TEXT_LEN = self.train_dataset.max_text_length
        weight = torch.ones(self.train_dataset.annotations_language.n_words).to(self.device)

        self.model  = self.model.to(self.device)


        criterion = nn.BCELoss()
        batch_index = 0        

        if not from_zero: self.load_latest_snapshot()

        batch_generator = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=32, shuffle=True, num_workers=1
        )

        loss_history = []


        for i in tqdm(range(n_epochs), desc='Training'):
        	
            self.model.train(True)

            loss_epoch = []
        
            for j, b in enumerate(tqdm(batch_generator, desc=f'Epoch {i+1} of {n_epochs}')):
          
                self.optimizer.zero_grad()
                self.model.zero_grad()

                in_ = b.to(self.device)

                mask = create_look_ahead_mask(in_)

                target = F.one_hot(in_.clone().detach()[:, 1:], num_classes=1186).float()
                out    = F.softmax(self.model(in_, mask), dim=-1)
            
                loss_value = 1. / 8. * criterion(out[:, :MAX_TEXT_LEN], target[:, :MAX_TEXT_LEN]) +\
                             7. / 8. * criterion(out[:, MAX_TEXT_LEN:-1], target[:, MAX_TEXT_LEN:])   
            	
                loss_value.backward()
                self.optimizer.step()

                batch_index += 1

                if batch_index % save_interval == 0: self.save_model()

                loss_epoch.append(loss_value.item())

        
            loss_history.append(np.mean(np.array(loss_epoch)))

            if plot_loss_history:
                plt.figure(figsize=(8, 8))
                plt.plot(loss_history, label='loss')
                plt.legend()
                plt.show()

        self.model = self.model.cpu()

        return loss_history



