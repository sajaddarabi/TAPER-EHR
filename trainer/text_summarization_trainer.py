import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.gru_ae import *
import random

class TextSummTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(TextSummTrainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer'].get('log_step', int(np.sqrt(data_loader.batch_size)))


    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        self.model.encoder.set_device(self.device)
        self.model.decoder.set_device(self.device)

        for batch_idx, data in enumerate(self.data_loader):
            x, i_s, b_is = list(map(lambda x: x.to(self.device), data))

            #x, i_s, b_is = x.to(self.device), i_s.to(self.device), b_is.to(self.device)

            self.optimizer.zero_grad()
            encoder_outputs = torch.zeros(x.shape[0], self.model.encoder.hidden_size, device=self.device)
            encoder_hidden = self.model.encoder.init_hidden(x.shape[1])

            encoder_outputs = self.model.encoder(x, encoder_hidden, i_s, b_is)
            decoder_hidden = self.model.decoder.init_hidden(x.shape[1])
            decoder_output = self.model.decoder(encoder_outputs, decoder_hidden)

            #use_teacher_forcing = True if random.random() < self.model.teacher_forcing_ratio else False


            #if use_teacher_forcing:
            #    # Teacher forcing: Feed the target as the next input
            #    for di in range(target_length):
            #        decoder_output, decoder_hidden, attn_weights = self.model.decoder(
            #                decoder_input, decoder_hidden, encoder_outputs)
            #        loss += self.loss(decoder_output, x[di])
            #        decoder_input = x[di]#encoder_outputs[di]  # Teacher forcing

            #else:
            #    # Without teacher forcing: use its own predictions as the next input
            #    for di in range(target_length):
            #        decoder_output, decoder_hidden, attn_weights = self.model.decoder(
            #                decoder_input, decoder_hidden, encoder_outputs)
            #        #topv, topi = decoder_output.topk(1)
            #        decoder_input = decoder_output.detach()#decoder_output.detach()#topi.squeeze().detach()  # detach from history as input
            #        loss += self.loss(decoder_output, x[di])
            loss = self.loss(decoder_output, x.detach(), i_s.detach())
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            loss = loss.detach()
            total_loss += loss

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] {}: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    'loss', loss))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': total_metrics / len(self.data_loader),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                x, i_s, b_is = list(map(lambda x: x.to(self.device), data))
                #x, i_s, b_is = x.to(self.device), i_s.to(self.device), b_is.to(self.device)
                #encoder_optimizer.zero_grad()
                #decoder_optimizer.zero_grad()
                #input_length = i_s.item()
                #target_length = input_length #target_tensor.size(0)
                                             #max length x.shape[0]

                encoder_outputs = torch.zeros(x.shape[0], self.model.encoder.hidden_size, device=self.device)

                self.model.encoder.set_device(self.device)
                self.model.decoder.set_device(self.device)


                encoder_hidden = self.model.encoder.init_hidden(x.shape[1])
                encoder_outputs = self.model.encoder(x, encoder_hidden, i_s, b_is)

                decoder_input = torch.zeros_like(x[0])#encoder_outputs[ei]

                decoder_hidden = self.model.decoder.init_hidden(x.shape[1])
                output = self.model.decoder(encoder_outputs, decoder_hidden)

                #for ei in range(input_length):
                #    encoder_output, encoder_hidden = self.model.encoder(
                #            x[ei], encoder_hidden)
                #    encoder_outputs[ei] = encoder_output[0, 0]
                #decoder_input = torch.zeros_like(x[0])#encoder_hidden #encoder_outputs[ei]
                #decoder_hidden = self.model.decoder.init_hidden()

                ## Without teacher forcing: use its own predictions as the next input
                #for di in range(target_length):
                #    decoder_output, decoder_hidden, _ = self.model.decoder(
                #            decoder_input, decoder_hidden, encoder_outputs)
                #    #topv, topi = decoder_output.topk(1)
                #    decoder_input = decoder_output.detach()#decoder_output.detach()#topi.squeeze().detach()  # detach from history as input
                    #loss += self.loss(decoder_output, x[di])
                #loss = (1.0 / i_s.item()) * loss
                loss = self.loss(output, x, i_s)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
