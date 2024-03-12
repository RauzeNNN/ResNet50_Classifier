import copy
import os
import time
from tqdm import tqdm
import torch
from loss import calc_loss
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, model, dtype, device, output_save_dir, dataloaders, batch_size, optimizer, patience, num_epochs, loss_function, accuracy_metric,  lr_scheduler=None, start_epoch=1):
        self.model = model
        self.dataloader = dataloaders
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr_scheduler = lr_scheduler
        self.best_loss = 1e9
        self.phases = ["train", "val"]
        self.best_model = []
        self.best_val_score = 0
        self.best_train_acc = 0
        self.best_val_loss = 0
        self.best_train_loss = 0
        self.batch_size = batch_size
        self.output_save_dir = output_save_dir
        self.dtype = dtype
        self.device = device
        self.loss_function = loss_function
        self.accuracy_metric = accuracy_metric
        self.train_loss_list = []
        self.val_loss_list = []

    def plot_loss_functions(self,name):
        plt.figure(figsize=(8, 4))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(np.arange(len(self.train_loss_list)),
                 self.train_loss_list, label='train loss')
        plt.plot(np.arange(len(self.val_loss_list)),
                 self.val_loss_list, label='val loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_save_dir, '{}.png'.format(name)))
        plt.cla()

    def train(self):
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")

        file = open(log_file, 'a')
        total_memory = f'{torch.cuda.get_device_properties(0).total_memory/ 1E9 if torch.cuda.is_available() else 0:.3g}G'

        for epoch in range(self.start_epoch, self.num_epochs+1):
            # print('Epoch {}/{}'.format(epoch, self.num_epochs))
            # print('-' * 10)
            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in self.phases:
                epoch_loss = 0.0
                correct_prediction = 0.0
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("LR", param_group['lr'])
                        file.write(f"LR {param_group['lr']}")
                        file.write("\n")
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                batch_step = 0
                with tqdm(self.dataloader[phase], unit="batch") as tbar:
                    for input_imgs, gt_labels, input_imgs_paths in tbar:
                        tbar.set_description(f"Epoch {epoch}")
                        batch_step += 1
                        input_imgs = input_imgs.to(self.device).type(self.dtype)
                        gt_labels = gt_labels.to(self.device).type(self.dtype)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            logits = self.model(input_imgs)
                            softmaxed_scores = self.model.softmax(logits)
                            _, predictions = torch.max(softmaxed_scores, 1)
                            _, gt_labels = torch.max(gt_labels, 1)

                            loss = calc_loss(softmaxed_scores.float(), gt_labels,
                                            loss_type=self.loss_function)
                            reserved = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            mem = reserved + '/' + total_memory
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                epoch_loss += loss.item()
                                correct_prediction += torch.sum(
                                    predictions == gt_labels).item()
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=100. * (correct_prediction/(batch_step*self.batch_size)), memory=mem)

                            else:
                                epoch_loss += loss.item()
                                correct_prediction += torch.sum(
                                    predictions == gt_labels).item()
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=100. * (correct_prediction/(batch_step*self.batch_size)), memory=mem)

                epoch_loss /= batch_step
                # deep copy the model
                if phase == 'val':
                    val_score = correct_prediction /max(len(self.dataloader[phase].dataset), 1)
                    if self.lr_scheduler:
                        # lr_scheduler.step(epoch_loss)
                        self.lr_scheduler.step(val_score)

                    self.val_loss_list.append(epoch_loss)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_score))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_score}"))

                    file.write("\n")
                    if epoch_loss < self.best_loss:
                        self.best_loss = epoch_loss
                        print("saving best model")
                        file.write("saving best model")
                        file.write("\n")
                        self.best_loss = epoch_loss
                        self.best_model = copy.deepcopy(
                            self.model.state_dict())
                        model_name = 'epoch{}.pt'.format(epoch)
                        save_dir = os.path.join(self.output_save_dir, 'models/')
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(self.best_model, os.path.join(
                            save_dir, model_name))
                else:
                    self.train_loss_list.append(epoch_loss)
                    print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                    file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                    file.write("\n")

            torch.save(self.model.state_dict(), os.path.join(
                save_dir, 'last_epoch.pt'))

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_score))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_score))

        file.write("\n")
        file.close()
        # load best model weights
        self.model.load_state_dict(self.best_model)
        self.plot_loss_functions('total')

        return self.model
