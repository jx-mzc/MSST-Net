from metrics import calc_psnr, calc_ssim
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import visdom


class Trainer():
    def __init__(self, args, logger, dataloader, model, device, optimizer):
        self.viz = visdom.Visdom(env='cave')
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.criterion = nn.L1Loss()

        self.optimizer = optimizer
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            # model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k: v for k, v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0):
        self.model.train()
        self.logger.info('Current epoch learning rate: %e' % (self.optimizer.param_groups[0]['lr']))
        train_loss = []

        for i_batch, sample_batched in enumerate(tqdm(self.dataloader['train'])):
            sample_batched = self.prepare(sample_batched)
            hrhsi = sample_batched['hrhsi']
            hrmsi = sample_batched['hrmsi']
            lrhsi = sample_batched['lrhsi']

            sr = self.model(lrhsi, hrmsi)
            loss = self.criterion(sr, hrhsi)
            loss.requires_grad_(True)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())

        self.logger.info('epoch: ' + str(current_epoch))
        self.logger.info('train loss: %.10f' % (np.mean(train_loss)))
        self.viz.line(X=np.array([current_epoch]), Y=np.array([np.mean(train_loss)]), win='tarin_loss_win',
                      opts={'title': 'train_loss', 'legend': ['train']},
                      update=None if current_epoch == 1 else 'append')

        if (current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.', ''): tmp[key] for key in tmp if
                                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/') + '/model/model_' + str(current_epoch).zfill(5) + '.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
        eval_loss = []
        psnr = []
        ssim = []

        self.model.eval()
        with torch.no_grad():

            for i_batch, sample_batched in enumerate(tqdm(self.dataloader['eval'])):
                sample_batched = self.prepare(sample_batched)
                hrhsi = sample_batched['hrhsi']
                hrmsi = sample_batched['hrmsi']
                lrhsi = sample_batched['lrhsi']

                with torch.no_grad():
                    sr = self.model(lrhsi, hrmsi)
                loss = self.criterion(sr, hrhsi)
                eval_loss.append(loss.item())

                sr = torch.clamp(sr, 0.0, 1.0)

                psnr.append(calc_psnr(hrhsi.detach(), sr.detach()))
                ssim.append(calc_ssim(hrhsi.detach(), sr.detach()))

            psnr_ave = np.mean(psnr)
            ssim_ave = np.mean(ssim)

            self.logger.info('eval loss: %.10f' % (np.mean(eval_loss)))
            self.logger.info('Eval PSNR (now): %.6f \t SSIM (now): %.6f' % (psnr_ave, ssim_ave))
            if (psnr_ave > self.max_psnr):
                self.max_psnr = psnr_ave
                self.max_psnr_epoch = current_epoch
            if (ssim_ave > self.max_ssim):
                self.max_ssim = ssim_ave
                self.max_ssim_epoch = current_epoch
            self.logger.info('Eval  PSNR (max): %.6f (%d) \t SSIM (max): %.6f (%d)'
                                % (self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

            self.viz.line(X=np.array([current_epoch]), Y=np.array([np.mean(eval_loss)]), win='eval_loss_win',
                              opts={'title': 'eval_loss', 'legend': ['eval']},
                              update=None if current_epoch == self.args.val_every else 'append')
            self.viz.line(X=np.array([current_epoch]), Y=np.array([psnr_ave]), win='psnr_win',
                              opts={'title': 'psnr', 'legend': ['psnr']},
                              update=None if current_epoch == self.args.val_every else 'append')
            self.viz.line(X=np.array([current_epoch]), Y=np.array([ssim_ave]), win='ssim_win',
                              opts={'title': 'ssim', 'legend': ['ssim']},
                              update=None if current_epoch == self.args.val_every else 'append')
        self.logger.info('Evaluation over.')
