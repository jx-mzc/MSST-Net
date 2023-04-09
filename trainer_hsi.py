import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import visdom


class Trainer():
    def __init__(self, args, logger, dataloader, model, device, optimizer):
        self.viz = visdom.Visdom(env='cave_hsi')
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0):
        self.model.train()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))
        train_loss = []

        for i_batch, sample_batched in enumerate(tqdm(self.dataloader['train'])):

            sample_batched = self.prepare(sample_batched)
            lrhsi = sample_batched['lrhsi']

            outputs = self.model(lrhsi)
            loss = self.criterion(outputs, lrhsi)
            loss.requires_grad_(True)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())

        self.logger.info('epoch: ' + str(current_epoch))
        self.logger.info('train loss: %.10f' % (np.mean(train_loss)))
        self.viz.line(X=np.array([current_epoch]), Y=np.array([np.mean(train_loss)]), win='hsi_tarin_loss_win',
                      opts={'title': 'hsi_train_loss', 'legend': ['hsi_train']},
                      update=None if current_epoch == 1 else 'append')

        if (current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
        eval_loss = []
        self.model.eval()
        with torch.no_grad():

            for i_batch, sample_batched in enumerate(tqdm(self.dataloader['eval'])):
                sample_batched = self.prepare(sample_batched)
                sample_batched = self.prepare(sample_batched)
                lrhsi = sample_batched['lrhsi']
                with torch.no_grad():
                    outputs = self.model(lrhsi)

                ### calculate evaluate loss
                loss = self.criterion(outputs, lrhsi)
                eval_loss.append(loss.item())

            self.logger.info('eval loss: %.10f' % (np.mean(eval_loss)))
            self.viz.line(X=np.array([current_epoch]), Y=np.array([np.mean(eval_loss)]), win='hsi_eval_loss_win',
                          opts={'title': 'hsi_eval_loss', 'legend': ['hsi_eval']},
                          update=None if current_epoch == self.args.val_every else 'append')
        self.logger.info('Evaluation over.')
