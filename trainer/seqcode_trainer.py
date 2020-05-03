import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.transformer_utils import init_weights

class SeqCodeTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(SeqCodeTrainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer'].get('log_step', int(np.sqrt(data_loader.batch_size)))
        self.loss = loss
        wi = lambda x: init_weights.weights_init(x, config=self.config['model'])
        self.model.apply(init_weights.weights_init)

        self.code_loss = config['trainer'].get('code_loss', False)
        self.visit_loss = config['trainer'].get('visit_loss', False)

    def _eval_metrics(self, output, target, **kwargs):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target, **kwargs)
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
        for batch_idx, (x, m, ivec, jvec, demo) in enumerate(self.data_loader):
            data, mask, ivec, jvec, demo = x.to(self.device), m.to(self.device), ivec.to(self.device), jvec.to(self.device), demo.to(self.device)

            target = data

            self.optimizer.zero_grad()
            logits, emb_w = self.model(data.float(), target, target_mask=mask, demo=demo)

            target = target.transpose(1, 0).contiguous().view(-1, target.size(-1)).float()
            mask = mask.transpose(1, 0).contiguous().view(-1, 1)
            logits = logits * mask

            loss_dict = self.loss(target, mask, logits, self.model.loss, emb_w, ivec, jvec, window=self.config["loss_window"])
            loss = loss_dict['visit_loss'] * self.visit_loss + loss_dict['code_loss'] * self.code_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip', 0.25))

            self.optimizer.step()
            loss = loss.detach()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_metrics += self._eval_metrics(logits.detach(), target.detach(), mask=mask.bool().detach())#, k=self.config['trainer']['recall_k'])

            if self.verbosity >= 2 and (batch_idx % self.log_step == 0):
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] {}: {:.6f}, {}: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    'visit_loss', loss_dict['visit_loss'],
                    'code_loss', loss_dict['code_loss']))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            total_loss += loss#loss_dict['visit_loss'] + loss_dict['code_loss']

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
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
            for batch_idx, (x, m, ivec, jvec, demo) in enumerate(self.valid_data_loader):
                data, mask, ivec, jvec, demo = x.to(self.device), m.to(self.device), ivec.to(self.device), jvec.to(self.device), demo.to(self.device)
                target = data

                logits, emb_w = self.model(data.float(), target, target_mask=mask, demo=demo)
                target = target.transpose(1, 0).contiguous().view(-1, target.size(-1)).float()
                mask = mask.transpose(1, 0).contiguous().view(-1, 1)
                logits = logits * mask

                loss_dict = self.loss(target, mask, logits, self.model.loss, emb_w, ivec, jvec, window=self.config["loss_window"])
                loss = loss_dict['visit_loss'] #+ loss_dict['code_loss']
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(logits.detach(), target.detach(), mask=mask.bool().detach())
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
