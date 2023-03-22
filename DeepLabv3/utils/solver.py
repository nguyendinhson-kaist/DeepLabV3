import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import copy
import os
from torch.utils.tensorboard import SummaryWriter

from utils.metric import SemanticMetric

# import matplotlib
# matplotlib.use('TkAgg')

# import matplotlib.pyplot as plt

class Solver(object):
    """ Class provide a solver instance to train or evaluate a model

    Required arguments:
    - model: nn.Model
    - train_loader: loader of training dataset
    - val_loader: loader of val dataset (use for evaluation)
    - num_classes: number of classes

    Optional arguments:
    - num_epochs: number of epochs to run for training. Default: 1
    - lr: learning rate used for optimizer. Default: 0.001
    - device: 'cpu' or 'cuda'. Default: 'cpu'
    - dtype: default: float 32
    - log_every: num of iters to log value of loss. Default: 25
    - verbose: print loss in terminal.Default: True
    """
    def __init__(self, model, train_loader, val_loader, num_classes, **kwargs) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        # Unpack keyword arguments
        self.num_epochs = kwargs.pop('num_epochs', 1)
        self.lr = kwargs.pop('lr', 0.001)
        self.device = kwargs.pop('device', 'cpu')
        self.dtype = kwargs.pop('dtype', torch.float32)
        self.log_every = kwargs.pop('log_every', 25)
        self.verbose = kwargs.pop('verbose', True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.SGD(params=[
        #     {'params': self.model.classifier.parameters()},
        #     {'params': self.model.backbone.parameters(), 'lr': self.lr*0.1}
        # ], lr=self.lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=self.num_epochs, power=0.9)

        self.best_params = None
        self.loss_history = []
        self.best_acc = 0
        self.best_iou = None

        self.run_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') \
            + '_lr' + str(self.lr) \
            + '_e' + str(self.num_epochs)

        # create logger
        self.logger = SummaryWriter('runs/dlv3_'+self.run_time)

        # evaluation metric
        self.metric = SemanticMetric(self.num_classes)

    def train(self):
        self.model.to(device=self.device)

        for e in range(self.num_epochs):
            self.logger.add_scalar('learning rate', self.scheduler.get_last_lr()[0], e)
            print('Lr: %f' % (self.scheduler.get_last_lr()[0]))

            for t, (X, y) in enumerate(self.train_loader):
                self.model.train()
                X = X.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=torch.long)

                scores = self.model(X)['out']
                
                void_mask = y < self.num_classes # only calculate loss for valid classes
                scores = scores.transpose(0, 1)[:, void_mask]

                loss = F.cross_entropy(scores.transpose(0,1), y[void_mask])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.loss_history.append(loss.item())

                if t % self.log_every == 0:
                    if self.verbose:
                        print('Iteration %d, loss= %.4f' % (t, loss.item()))
                        print()
                        
                    self.logger.add_scalar('training loss', loss.item(), e*len(self.train_loader)+t)

            mIoU, iou = self.check_accuracy()
            self.logger.add_scalar('mIoU', mIoU, e)
            print('Epoch: %d, mIoU = %f' % (e, mIoU*100))

            if mIoU > self.best_acc:
                self.best_acc = mIoU
                self.best_iou = iou
                self.best_params = copy.deepcopy(self.model.state_dict())

            self.scheduler.step()

        self.model.load_state_dict(self.best_params)
        print('Training finished, best mIoU: %f' % (self.best_acc*100))
        print('IoU classes: ', self.best_iou)
        self.logger.close()

        # create new folder
        folder = 'out/result_'+self.run_time

        if not os.path.exists(folder):
            os.mkdir(folder)

        torch.save(self.model, 'out/result_'+self.run_time+'/best_model.pth')
        torch.save(self.best_params, 'out/result_'+self.run_time+'/best_params.pth')

    def check_accuracy(self):
        self.model.to(self.device)
        self.model.eval()
        self.metric.reset()

        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(device=self.device, dtype=self.dtype)
                y = y.cpu().numpy().astype(int)

                pred = self.model(X)['out'].detach().argmax(1).cpu().numpy().astype(int)

                # update metric
                self.metric.update(y.flatten(), pred.flatten())

            mIoU, iou = self.metric.get_results()

        return mIoU, iou.tolist()
    
    def visualize_model(self):
        dataiter = iter(self.train_loader)
        imgs, lbls = next(dataiter)   
        self.logger.add_graph(self.model, imgs, use_strict_trace=False)