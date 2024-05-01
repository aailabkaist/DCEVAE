import os
import torch
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from torch import nn

def test(test_loader, args, logger):
    device = args.device
    model_path = os.path.join(args.save_path, 'model.pth')
    test_model = torch.load(model_path)
    test_model.to(device)
    test_model.eval()
    correct, _all, o1s, o2s, o3s, o4s, o1s_bin, o2s_bin, o3s_bin, o4s_bin, ys, ys_bin = \
        0, 0, None, None, None, None, None, None, None, None, None, None
    with torch.no_grad():
        for idx, (r, d, a, y, r2, d2, a2, y2) in enumerate(test_loader):
            loss_val, x_recon_loss_val, y_recon_loss_val, y_p_val, y_p_counter_val, u_kl_loss_val, vae_tc_loss_val, \
            d_tc_loss_val, fair_loss_val = test_model.calculate_loss(r.to(device), d.to(device), a.to(device), \
            y.to(device), r2.to(device), d2.to(device), a2.to(device), y2.to(device))

            # For saving the result:
            if args.tSNE == True:
                u_mu, u_logvar = test_model.q_u(r.to(device), d.to(device), a.to(device), y.to(device))
                u_prev = test_model.reparameterize(u_mu, u_logvar)
                ur_prev, ud_prev = torch.split(u_prev, [args.ur_dim, args.ud_dim], 1)
                ur = torch.cat((ur, ur_prev), 0) if idx != 0 else ur_prev
                ud = torch.cat((ud, ud_prev), 0) if idx != 0 else ud_prev
                a_all = torch.cat((a_all, a), 0) if idx != 0 else a

            y_p_val = nn.Sigmoid()(y_p_val)
            y_p_counter_val = nn.Sigmoid()(y_p_counter_val)
            label_predicted = torch.eq(y_p_val.gt(0.5).byte(), y.to(device).byte())
            correct += torch.sum(label_predicted)
            _all += float(label_predicted.size(0))

            y_p_np = y_p_val.cpu().detach().numpy()
            y_cf_np = y_p_counter_val.cpu().detach().numpy()
            mask_a = np.where(a == 1, -1, 1)
            cf_effect = (y_cf_np - y_p_np) * mask_a
            cf_bin = (np.greater(y_cf_np, 0.5).astype(int) - np.greater(y_p_np, 0.5).astype(int)) * mask_a
            
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

            o1_bin = cf_bin[mask1 == [True]]
            o2_bin = cf_bin[mask2 == [True]]
            o3_bin = cf_bin[mask3 == [True]]
            o4_bin = cf_bin[mask4 == [True]]

            o1s_bin = np.concatenate((o1s_bin, o1_bin), axis=0) if idx != 0 else o1_bin
            o2s_bin = np.concatenate((o2s_bin, o2_bin), axis=0) if idx != 0 else o2_bin
            o3s_bin = np.concatenate((o3s_bin, o3_bin), axis=0) if idx != 0 else o3_bin
            o4s_bin = np.concatenate((o4s_bin, o4_bin), axis=0) if idx != 0 else o4_bin

            ys = np.concatenate((ys, cf_effect), axis=0) if idx != 0 else cf_effect
            ys_bin = np.concatenate((ys_bin, cf_bin), axis=0) if idx != 0 else cf_bin

        if args.tSNE == True and args.ur_dim == 2:
            draw_2dim(ur, a_all, args, 'Ur')
        if args.tSNE == True and args.ud_dim == 2:
            draw_2dim(ud, a_all, args, 'Ud')

        if args.tSNE == True:
            draw_tSNE(ur, a_all, args, 'Ur')
            draw_tSNE(ud, a_all, args, 'Ud')

        logger.info('***data***')
        logger.info('cf: {:.4f}'.format(np.sum(ys) / ys.shape[0]))
        logger.info('o1: {:.8f}'.format(np.sum(o1s) / o1s.shape[0]))
        logger.info('o2: {:.8f}'.format(np.sum(o2s) / o2s.shape[0]))
        logger.info('o3: {:.8f}'.format(np.sum(o3s) / o3s.shape[0]))
        logger.info('o4: {:.8f}'.format(np.sum(o4s) / o4s.shape[0]))

        line = '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(np.sum(ys_bin) / ys_bin.shape[0], \
                                                                 np.sum(o1s) / o1s.shape[0], np.sum(o2s) / o2s.shape[0],
                                                                 np.sum(o3s) / o3s.shape[0], np.sum(o4s) / o4s.shape[0])
        file_dir = os.path.abspath(os.path.join(args.save_path, os.pardir))
        file_dir = os.path.join(file_dir, 'whole_log.txt')
        if not os.path.exists(file_dir):
            f = open(file_dir, 'w')
        else:
            f = open(file_dir, 'a')
        f.write('a_r_{:s}_a_d_{:s}_a_y_{:s}_a_h_{:s}_a_f_{:s}_a_a_{:s}_u_{:s}_ur_{:d}_ud_{:d}_run_{:d}'\
                                  .format(str(args.a_r), str(args.a_d), str(args.a_y), str(args.a_h), str(args.a_f), \
                                          str(args.a_a), str(args.u_kl), args.ur_dim, args.ud_dim, args.run))
        f.write(line)
        f.close()

def generate_data(loader, args, dataset='train'):
    device = args.device
    model_path = os.path.join(args.save_path, 'model.pth')
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, (r, d, a, y, r2, d2, a2, y2) in enumerate(loader):
            u_mu, u_logvar = model.q_u(r.to(device), d.to(device), a.to(device), y.to(device))
            u_prev = model.reparameterize(u_mu, u_logvar)
            r_hard, d_hard, d_cf_hard, y_hard, y_cf_hard = model.reconstruct_hard(u_prev, a.to(device))
            i_hard = torch.cat((r_hard, d_hard), 1)
            i_cf_hard = torch.cat((r_hard, d_cf_hard), 1)

            i_f = torch.cat((i_f, i_hard), 0) if idx != 0 else i_hard
            i_cf = torch.cat((i_cf, i_cf_hard), 0) if idx != 0 else i_cf_hard
            y_f = torch.cat((y_f, y_hard), 0) if idx != 0 else y_hard
            y_cf = torch.cat((y_cf, y_cf_hard), 0) if idx != 0 else y_cf_hard
            a_all = torch.cat((a_all, a), 0) if idx != 0 else a
            y_real = torch.cat((y_real, y), 0) if idx != 0 else y

        i_f = i_f.cpu().detach().numpy()
        i_cf = i_cf.cpu().detach().numpy()
        y_f = y_f.cpu().detach().numpy()
        y_cf = y_cf.cpu().detach().numpy()
        a_all = a_all.cpu().detach().numpy()
        a_cf_all = np.where(a_all == 0, 1, 0)
        y_real = y_real.cpu().detach().numpy()

    f_out_np = os.path.join(args.save_path, dataset)
    np.savez(f_out_np, input=i_f, input_cf=i_cf, y=y_f, y_cf=y_cf, a=a_all, a_cf=a_cf_all)

    print('Summary')
    print('Ground Truth, p(y=1):', np.sum(y_real) / y_real.shape[0])
    print('              p(y=1):', np.sum(y_f)/y_f.shape[0])
    print('           p(y_cf=1):', np.sum(y_cf)/y_cf.shape[0])


def draw_2dim(input, a, args, latent_name):
    input = input.cpu().detach().numpy()
    a = a.cpu().detach().numpy()
    colors = 'orange', 'm'  # , 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for c, label in zip(colors, [0.0, 1.0]):
        name = 'a=1' if label == 1.0 else 'a=0'
        index = np.where(a == label)
        plt.scatter(input[index, 0], input[index, 1], c=c, label=name, alpha=0.5)
    plt.title(latent_name + ' with respect to A')
    plt.legend()
    figfile = os.path.join(args.save_path, '2dim_'+latent_name + '_wrt_A')

    plt.savefig(figfile)

    plt.close()


def draw_tSNE(input, a, args, latent_name):
    from sklearn.manifold import TSNE
    import time
    t0 = time.time()
    print('tSNE start for ' + latent_name)

    input = input.cpu().detach().numpy()
    a = a.cpu().detach().numpy()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(input)

    colors = 'orange', 'm'  # , 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for c, label in zip(colors, [0.0, 1.0]):
        name = 'a=1' if label == 1.0 else 'a=0'
        index = np.where(a == label)
        print(index)
        plt.scatter(tsne_result[index, 0], tsne_result[index, 1], c=c, label=name, alpha=0.5)
    plt.title(latent_name + ' with respect to A')
    plt.legend()
    figfile = os.path.join(args.save_path, 'tSNE_' + latent_name + '_wrt_A')

    plt.savefig(figfile)

    plt.close()

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - t0))
