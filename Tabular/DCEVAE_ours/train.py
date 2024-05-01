import torch
import torch.optim as optim
import time
from tqdm import trange
import numpy as np
import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from torch import nn

def train(model, train_loader, valid_loader, args, logger):
    device = args.device
    model.to(device)

    model = model.train()

    """parameter selection for multi-step optimization"""
    params_without_delta = [param for name, param in model.named_parameters() if 'discriminator' not in name]
    params_delta = [param for name, param in model.named_parameters() if 'discriminator' in name]
    opt_without_delta = optim.Adam(params_without_delta, lr=args.lr)
    opt_delta = optim.Adam(params_delta, lr=args.lr)

    train_x_recon_losses = []
    train_y_recon_losses = []
    train_u_kl_losses = []
    train_vae_tc_losses = []
    train_d_tc_losses = []
    train_fair_losses = []

    valid_x_recon_losses = []
    valid_y_recon_losses = []
    valid_u_kl_losses = []
    valid_vae_tc_losses = []
    valid_d_tc_losses = []
    valid_fair_losses = []

    loss_log = []
    loss_val_log = []
    epoch_log = []
    acc_log = []

    best_acc = 0
    best_epoch = 0
    best_epoch_for_acc = 0
    best_epoch_for_loss = 0
    best_loss = 1e20
    start_time = time.time()

    for epoch_i in trange(args.n_epochs):
        model.train()
        correct, _all, o1s, o2s, o3s, o4s, o1s_bin, o2s_bin, o3s_bin, o4s_bin, ys, ys_bin, loss_whole =\
            0, 0, None, None, None, None, None, None, None, None, None, None, 0
        for idx, (r, d, a, y, r2, d2, a2, y2) in enumerate(train_loader):
            loss_val, x_recon_loss_val, y_recon_loss_val, y_p_val, y_p_counter_val, u_kl_loss_val, vae_tc_loss_val, \
            D_tc_loss_val, fair_loss_val = model.calculate_loss(r.to(device), d.to(device), a.to(device), \
            y.to(device), r2.to(device), d2.to(device), a2.to(device), y2.to(device))  # (*cur_batch)

            # optimization could be done more precisely but less efficiently by only updating delta or other params on a batch

            opt_delta.zero_grad()
            loss_max = D_tc_loss_val
            loss_max.backward(retain_graph=True)
            opt_delta.step()

            opt_without_delta.zero_grad()
            loss_val.backward()
            opt_without_delta.step()

            train_x_recon_losses.append(x_recon_loss_val.item())
            train_y_recon_losses.append(y_recon_loss_val.item())
            train_u_kl_losses.append(u_kl_loss_val.item())
            train_vae_tc_losses.append(vae_tc_loss_val.item())
            train_d_tc_losses.append(D_tc_loss_val.item())
            train_fair_losses.append(fair_loss_val.item())

            y_p_val = nn.Sigmoid()(y_p_val)
            y_p_counter_val = nn.Sigmoid()(y_p_counter_val)
            label_predicted = torch.eq(y_p_val.gt(0.5).byte(), y.to(device).byte())
            correct += torch.sum(label_predicted)
            _all += float(label_predicted.size(0))

            y_p_np = y_p_val.cpu().detach().numpy()
            y_cf_np = y_p_counter_val.cpu().detach().numpy()
            mask_a = np.where(a == 1, -1, 1)
            cf_effect = (y_cf_np - y_p_np) * mask_a
            # cf_bin = (np.greater(y_cf_np, 0.5).astype(int) - np.greater(y_p_np, 0.5).astype(int)) * mask_a

            m = r.cpu().detach().numpy()[:, 1:]

            mask1 = (m == [False, False]).all(axis=1)
            mask2 = (m == [False, True]).all(axis=1)
            mask3 = (m == [True, False]).all(axis=1)
            mask4 = (m == [True, True]).all(axis=1)

            o1 = cf_effect[mask1 == [True]]
            o2 = cf_effect[mask2 == [True]]
            o3 = cf_effect[mask3 == [True]]
            o4 = cf_effect[mask4 == [True]]

            o1s = np.concatenate((o1s, o1), axis=0) if idx != 0 else o1
            o2s = np.concatenate((o2s, o2), axis=0) if idx != 0 else o2
            o3s = np.concatenate((o3s, o3), axis=0) if idx != 0 else o3
            o4s = np.concatenate((o4s, o4), axis=0) if idx != 0 else o4

            ys = np.concatenate((ys, cf_effect), axis=0) if idx != 0 else cf_effect

            loss_whole += loss_val.cpu().detach().numpy()

        epoch_log.append(epoch_i)

        acc = correct.cpu().detach().numpy() / _all

        logger.info('Epoch {:d}'.format(epoch_i))

        logger.info('###Train###')

        logger.info('BCE(x): {:.4f}'.format(np.array(train_x_recon_losses[-len(train_loader):]).mean()))
        logger.info('BCE(y): {:.4f}'.format(np.array(train_y_recon_losses[-len(train_loader):]).mean()))
        logger.info('KL(u): {:.4f}'.format(np.array(train_u_kl_losses[-len(train_loader):]).mean()))
        logger.info('VAE tc: {:.4f}'.format(np.array(train_vae_tc_losses[-len(train_loader):]).mean()))
        logger.info('D tc: {:.4f}'.format(np.array(train_d_tc_losses[-len(train_loader):]).mean()))
        logger.info('fair_loss: {:.4f}'.format(np.array(train_fair_losses[-len(train_loader):]).mean()))

        logger.info('predict(y): {:.4f}'.format(acc))
        logger.info('cf: {:.4f}'.format(np.sum(ys) / ys.shape[0]))

        loss_log.append(loss_whole)
        acc_log.append(acc)

        model.eval()
        correct, _all, o1s, o2s, o3s, o4s, o1s_bin, o2s_bin, o3s_bin, o4s_bin, ys, ys_bin, loss_whole = \
            0, 0, None, None, None, None, None, None, None, None, None, None, 0
        with torch.no_grad():
            for idx, (r, d, a, y, r2, d2, a2, y2) in enumerate(valid_loader):
                loss_val, x_recon_loss_val, y_recon_loss_val, y_p_val, y_p_counter_val, u_kl_loss_val, vae_tc_loss, \
                D_tc_loss, fair_loss_val = model.calculate_loss(r.to(device), d.to(device), a.to(device), y.to(device),\
                r2.to(device), d2.to(device), a2.to(device), y2.to(device))  # (*cur_batch)

                valid_x_recon_losses.append(x_recon_loss_val.item())
                valid_y_recon_losses.append(y_recon_loss_val.item())
                valid_u_kl_losses.append(u_kl_loss_val.item())
                valid_vae_tc_losses.append(vae_tc_loss.item())
                valid_d_tc_losses.append(D_tc_loss.item())
                valid_fair_losses.append(fair_loss_val.item())

                y_p_val = nn.Sigmoid()(y_p_val)
                y_p_counter_val = nn.Sigmoid()(y_p_counter_val)
                label_predicted = torch.eq(y_p_val.gt(0.5).byte(), y.to(device).byte())
                correct += torch.sum(label_predicted)
                _all += float(label_predicted.size(0))

                y_p_np = y_p_val.cpu().detach().numpy()
                y_cf_np = y_p_counter_val.cpu().detach().numpy()
                mask_a = np.where(a == 1, -1, 1)
                cf_effect = (y_cf_np - y_p_np) * mask_a

                m = r.cpu().detach().numpy()[:, 1:]

                mask1 = (m == [False, False]).all(axis=1)
                mask2 = (m == [False, True]).all(axis=1)
                mask3 = (m == [True, False]).all(axis=1)
                mask4 = (m == [True, True]).all(axis=1)

                o1 = cf_effect[mask1 == [True]]
                o2 = cf_effect[mask2 == [True]]
                o3 = cf_effect[mask3 == [True]]
                o4 = cf_effect[mask4 == [True]]

                o1s = np.concatenate((o1s, o1), axis=0) if idx != 0 else o1
                o2s = np.concatenate((o2s, o2), axis=0) if idx != 0 else o2
                o3s = np.concatenate((o3s, o3), axis=0) if idx != 0 else o3
                o4s = np.concatenate((o4s, o4), axis=0) if idx != 0 else o4

                loss_whole += loss_val

                ys = np.concatenate((ys, cf_effect), axis=0) if idx != 0 else cf_effect

            loss_val_log.append(loss_whole)

            logger.info('\n###Valid###')

            logger.info('BCE(x): {:.4f}'.format(np.array(valid_x_recon_losses[-len(valid_loader):]).mean()))
            logger.info('BCE(y): {:.4f}'.format(np.array(valid_y_recon_losses[-len(valid_loader):]).mean()))
            logger.info('KL(s): {:.4f}'.format(np.array(valid_u_kl_losses[-len(valid_loader):]).mean()))
            logger.info('vae tc: {:.4f}'.format(np.array(valid_vae_tc_losses[-len(valid_loader):]).mean()))
            logger.info('D tc_loss: {:.4f}'.format(np.array(valid_d_tc_losses[-len(valid_loader):]).mean()))
            logger.info('fair_loss: {:.4f}'.format(np.array(valid_fair_losses[-len(valid_loader):]).mean()))

            logger.info('***data***')
            logger.info('cf: {:.4f}'.format(np.sum(ys) / ys.shape[0]))
            logger.info('o1: {:.8f}'.format(np.sum(o1s) / o1s.shape[0]))
            logger.info('o2: {:.8f}'.format(np.sum(o2s) / o2s.shape[0]))
            logger.info('o3: {:.8f}'.format(np.sum(o3s) / o3s.shape[0]))
            logger.info('o4: {:.8f}'.format(np.sum(o4s) / o4s.shape[0]))
            acc = correct.cpu().detach().numpy() /_all
            loss_check = loss_whole.item() / _all

            if epoch_i == 0 and loss_check > best_loss:
                best_loss = loss_check

            print('now best epoch is, best loss, loss_check', best_epoch, best_loss, loss_check)
            print('loss_check < best_loss', loss_check < best_loss)
            if acc > best_acc:
                if args.a_f == 0:
                    model_path = os.path.join(args.save_path, 'model.pth')
                    torch.save(model, model_path)
                    best_epoch = epoch_i
                    print('best epoch update by acc, epoch is ', epoch_i)
                best_epoch_for_acc = epoch_i
                best_acc = acc

            if loss_check < best_loss:
                if args.a_f != 0:
                    model_path = os.path.join(args.save_path, 'model.pth')
                    torch.save(model, model_path)
                    best_epoch = epoch_i
                    print('best epoch update by loss, epoch is ', epoch_i)
                best_epoch_for_loss = epoch_i
                best_loss = loss_check

            if epoch_i - best_epoch > args.break_epoch and args.early_stop == True:
                line = 'time elapsed: {:.4f}min'.format((time.time() - start_time) / 60.0)
                logger.info(line)
                break

        if args.early_stop == False:
            model_path = os.path.join(args.save_path, 'model.pth')
            torch.save(model, model_path)

        line = 'time elapsed: {:.4f}min'.format((time.time() - start_time) / 60.0)
        print(line)

    line = 'best epoch for loss is {:d}\n'.format(best_epoch_for_loss)
    line += 'best epoch for accuracy is {:d}'.format(best_epoch_for_acc)
    logger.info(line)

    fig, axs = plt.subplots(2)
    axs[0].set_title('loss')
    axs[0].plot(epoch_log, loss_log, '-r', label='loss')
    axs[0].plot(epoch_log, loss_val_log, '-b', label='validation loss')
    axs[1].set_title('acc')
    axs[1].plot(epoch_log, acc_log, '-r', label='accuracy')

    # plt.xlabel("n iteration")
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')

    # save image
    plt_name = 'loss_and_accuracy.png'
    plt_path = os.path.join(args.save_path, plt_name)
    plt.savefig(plt_path)  # should before show method
    plt.close()

    # show
    # plt.show()