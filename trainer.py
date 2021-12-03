import os
import time
import shutil
import pickle

# import torch
import paddle
# import torch.nn.functional as F
import paddle.nn.functional as F

from tqdm import tqdm
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from model import RecurrentAttention
from utils import AverageMeter


class Trainer:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config

        # if config.use_gpu and torch.cuda.is_available():
        if config.use_gpu and paddle.device.cuda.device_count():
            # self.device = torch.device("cuda")
            self.place = paddle.CUDAPlace(0)
        else:
            # self.device = torch.device("cpu")
            self.place = paddle.CPUPlace()

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            # self.num_train = len(self.train_loader.sampler.indices)
            # self.num_valid = len(self.valid_loader.sampler.indices)
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "ram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.glimpse_hidden,
            self.loc_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        )
        # self.model.to(self.device)
        self.model.to(device=self.place)

        # # initialize optimizer and scheduler
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.config.init_lr
        # )
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer, "min", patience=self.lr_patience
        # )

        # initialize optimizer and scheduler
        self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            learning_rate=self.config.init_lr, mode='min',
            patience=self.lr_patience, verbose=True
        )
        self.optimizer = paddle.optimizer.Adam(
            parameters=self.model.parameters(),
            learning_rate=self.scheduler,
        )

    def reset(self):
        # h_t = torch.zeros(
        #     self.batch_size,
        #     self.hidden_size,
        #     dtype=torch.float,
        #     device=self.device,
        #     requires_grad=True,
        # )
        h_t = paddle.zeros(
            shape=[self.batch_size, self.hidden_size],
            dtype="float32",
        )
        h_t = paddle.to_tensor(h_t, place=self.place, stop_gradient=False)
        l_t = paddle.uniform(shape=[self.batch_size, 2], dtype='float32', min=-1.0, max=1.0)
        l_t = paddle.to_tensor(l_t, place=self.place, stop_gradient=False)

        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(self.start_epoch, self.epochs):

            # print(
            #     "\nEpoch: {}/{} - LR: {:.6f}".format(
            #         epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
            #     )
            # )
            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.scheduler.last_lr
                )
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            # self.scheduler.step(-valid_acc)
            self.scheduler.step(valid_loss)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc
                )
            )

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                # self.optimizer.zero_grad()
                self.optimizer.clear_grad()

                # x, y = x.to(self.device), y.to(self.device)
                x = paddle.to_tensor(x, place=self.place, stop_gradient=True)
                y = paddle.to_tensor(y, place=self.place, stop_gradient=True)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                log_pi = []
                baselines = []
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t, last=False)

                    # store
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                # log_probas.shape = [batch_size, num_classes]
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                log_pi.append(p)
                baselines.append(b_t)
                locs.append(l_t[0:9])

                # # convert list to tensors and reshape
                # baselines = torch.stack(baselines).transpose(1, 0)
                # log_pi = torch.stack(log_pi).transpose(1, 0)

                # convert list to tensors and transpose,
                # after being transposed, baselines and log_pi have the shape
                # [batch_size, num_glimpses]
                baselines = paddle.stack(baselines, axis=0).transpose([1, 0])
                log_pi = paddle.stack(log_pi, axis=0).transpose([1, 0])

                # # calculate reward
                # predicted = torch.max(log_probas, 1)[1]
                # R = (predicted.detach() == y).float()
                # R = R.unsqueeze(1).repeat(1, self.num_glimpses)

                # calculate reward
                # predicted and y have shape [batch_size, 1]
                # R has shape [batch_size, num_glimpses], the elements of R are in {0.0, 1.0}
                predicted = paddle.argmax(log_probas, axis=1, keepdim=True)
                R = (predicted.detach() == y).astype('float32')
                R = R.tile(repeat_times=[1, self.num_glimpses])

                # compute losses for differentiable modules
                loss_action = F.nll_loss(log_probas, y.squeeze())
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                # loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
                # loss_reinforce = torch.mean(loss_reinforce, dim=0)
                loss_reinforce = paddle.sum(-log_pi * adjusted_reward, axis=1)
                loss_reinforce = paddle.mean(loss_reinforce, axis=0)

                # sum up into a hybrid loss
                loss = loss_action + loss_baseline + loss_reinforce * 0.01

                # # compute accuracy
                # correct = (predicted == y).float()
                # acc = 100 * (correct.sum() / len(y))

                # compute accuracy
                correct = (predicted == y).astype('float32')
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.item(), x.shape[0])
                accs.update(acc.item(), x.shape[0])

                # compute gradients and update parameters
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    # imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    # locs = [l.cpu().data.numpy() for l in locs]
                    imgs = [g.squeeze().numpy() for g in imgs]
                    locs = [l.numpy() for l in locs]
                    pickle.dump(
                        # imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb")
                        imgs, open(os.path.join(self.plot_dir, "g_{}.p".format(epoch + 1)), "wb")
                    )
                    pickle.dump(
                        # locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb")
                        locs, open(os.path.join(self.plot_dir, "l_{}.p".format(epoch + 1)), "wb")
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)

            # return losses.avg, accs.avg
        return losses.avg, accs.avg

    # @torch.no_grad()
    @paddle.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            # x, y = x.to(self.device), y.to(self.device)
            x = paddle.to_tensor(x, place=self.place, stop_gradient=True)
            y = paddle.to_tensor(y, place=self.place, stop_gradient=True)

            # duplicate M times
            # x = x.repeat(self.M, 1, 1, 1)
            x = x.tile(repeat_times=[self.M, 1, 1, 1])

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t, last=False)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
            log_pi.append(p)
            baselines.append(b_t)

            # # convert list to tensors and reshape
            # baselines = torch.stack(baselines).transpose(1, 0)
            # log_pi = torch.stack(log_pi).transpose(1, 0)

            # convert list to tensors and transpose,
            # after being transposed, baselines and log_pi have the shape
            # [batch_size, num_glimpses]
            baselines = paddle.stack(baselines, axis=0).transpose([1, 0])
            log_pi = paddle.stack(log_pi, axis=0).transpose([1, 0])

            # # average
            # log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            # log_probas = torch.mean(log_probas, dim=0)

            # average
            # the final log_probas has shape [original_batch_size, num_classes]
            log_probas = paddle.reshape(log_probas, shape=[self.M, -1, log_probas.shape[-1]])
            log_probas = paddle.mean(log_probas, axis=0)

            # baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            # baselines = torch.mean(baselines, dim=0)

            # the final "baselines" has shape [original_batch_size, num_glimpses]
            baselines = paddle.reshape(baselines, shape=[self.M, -1, baselines.shape[-1]])
            baselines = paddle.mean(baselines, axis=0)

            # log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            # log_pi = torch.mean(log_pi, dim=0)

            # the final "log_pi" has shape [original_batch_size, num_glimpses]
            log_pi = paddle.reshape(log_pi, shape=[self.M, -1, log_pi.shape[-1]])
            log_pi = paddle.mean(log_pi, axis=0)

            # # calculate reward
            # predicted = torch.max(log_probas, 1)[1]
            # R = (predicted.detach() == y).float()
            # R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # calculate reward
            # predicted and y have shape [original_batch_size, 1]
            # R has shape [original_batch_size, num_glimpses], the elements of R are in {0.0, 1.0}
            predicted = paddle.argmax(log_probas, axis=1, keepdim=True)
            R = (predicted.detach() == y).astype('float32')
            R = R.tile(repeat_times=[1, self.num_glimpses])

            # compute losses for differentiable modules
            # loss_action = F.nll_loss(log_probas, y)
            loss_action = F.nll_loss(log_probas, y.squeeze())
            loss_baseline = F.mse_loss(baselines, R)

            # # compute reinforce loss
            # adjusted_reward = R - baselines.detach()
            # loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            # loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = paddle.sum(-log_pi * adjusted_reward, axis=1)
            loss_reinforce = paddle.mean(loss_reinforce, axis=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce * 0.01

            # # compute accuracy
            # correct = (predicted == y).float()
            # acc = 100 * (correct.sum() / len(y))

            # compute accuracy
            correct = (predicted == y).astype('float32')
            acc = 100 * (correct.sum() / len(y))

            # # store
            # losses.update(loss.item(), x.size()[0])
            # accs.update(acc.item(), x.size()[0])

            # store
            losses.update(loss.item(), x.shape[0])
            accs.update(acc.item(), x.shape[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, accs.avg

    # @torch.no_grad()
    @paddle.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, y) in enumerate(self.test_loader):
            # x, y = x.to(self.device), y.to(self.device)
            x = paddle.to_tensor(x, place=self.place, stop_gradient=True)
            y = paddle.to_tensor(y, place=self.place, stop_gradient=True)

            # duplicate M times
            # x = x.repeat(self.M, 1, 1, 1)
            # self.M should be a positive integer, such as 1, 2, 3, ...
            x = x.tile(repeat_times=[self.M, 1, 1, 1])

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t, last=False)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)

            # log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            # log_probas = torch.mean(log_probas, dim=0)

            # the final "log_probas" has shape [original_batch_size, num_classes]
            log_probas = paddle.reshape(log_probas, shape=[self.M, -1, log_probas.shape[-1]])
            log_probas = paddle.mean(log_probas, axis=0)

            # pred = log_probas.data.max(1, keepdim=True)[1]
            # correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            pred = paddle.argmax(log_probas, axis=1, keepdim=True)
            correct += (pred == y).astype('float32').sum().item()

        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - Error: {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )

    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pdparams"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        # torch.save(state, ckpt_path)
        paddle.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + "_model_best.pdparams"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pdparams"
        if best:
            filename = self.model_name + "_model_best.pdparams"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        # ckpt = torch.load(ckpt_path)
        ckpt = paddle.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        # self.model.load_state_dict(ckpt["model_state"])
        self.model.set_state_dict(ckpt["model_state"])
        # self.optimizer.load_state_dict(ckpt["optim_state"])
        self.optimizer.set_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))


if __name__ == "__main__":
    import paddle

    batch_size = 5
    num_classes = 10
    num_glimpses = 6
    y = paddle.randint(low=0, high=9, shape=[batch_size, 1])
    log_probas = paddle.randn(shape=[batch_size, num_classes])
    predicted = paddle.argmax(log_probas, axis=1, keepdim=True)
    print(predicted)
    R = (predicted.detach() == y).astype('float32')
    print(R)
    R = R.tile(repeat_times=[1, num_glimpses])
    print(R)


