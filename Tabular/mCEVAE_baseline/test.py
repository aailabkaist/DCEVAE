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
        for idx, (x, o, a, y) in enumerate(test_loader):
            loss_val, xo_recon_loss_val, y_recon_loss_val, y_p_val, y_p_counter_val, mmd_loss_val, mmd_a_loss_val, \
            fair_loss_val = test_model.calculate_loss(x.to(device), o.to(device), a.to(device), y.to(device))  # (*cur_batch)

            # For saving the result:
            if args.tSNE == True or args.u_dim == 2:
                _, _, u_val, u0, u1 = test_model.forward(o.to(device), a.to(device), x.to(device))
                u = torch.cat((u, u_val), 0) if idx != 0 else u_val
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

            m = o.cpu().detach().numpy()
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

            if args.debug == True:
                y_p = y_p_val.gt(0.5).byte()
                y_real = torch.cat((y_real, y), 0) if idx != 0 else y
                y_pred = torch.cat((y_pred, y_p), 0) if idx != 0 else y_p

        if args.u_dim == 2:
            draw_2dim(u, a_all, args, 'U')

        if args.tSNE == True:
            draw_tSNE(u, a_all, args, 'U')

        logger.info('***data***')
        logger.info('cf: {:.4f}'.format(np.sum(ys) / ys.shape[0]))
        logger.info('o1: {:.8f}'.format(np.sum(o1s) / o1s.shape[0]))
        logger.info('o2: {:.8f}'.format(np.sum(o2s) / o2s.shape[0]))
        logger.info('o3: {:.8f}'.format(np.sum(o3s) / o3s.shape[0]))
        logger.info('o4: {:.8f}'.format(np.sum(o4s) / o4s.shape[0]))

        if args.debug == True:
            y_real = y_real.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            logger.info('Ground Truth P(Y=1): {:.4f}'.format(np.sum(y_real) / y_real.shape[0]))
            logger.info('Predicted    P(Y=1): {:.4f}' .format(np.sum(y_pred) / y_pred.shape[0]))

def generate_data(loader, args, dataset='train'):
    device = args.device
    model_path = os.path.join(args.save_path, 'model.pth')
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, (x, o, a, y) in enumerate(loader):
            u_mu_encoder, u_logvar_encoder, u, u0, u1 = model.forward(o.to(device), a.to(device), x.to(device))
            o_hard, o_cf_hard, x_hard, x_cf_hard, y_hard, y_cf_hard = model.reconstruct_hard(a.to(device), u)

            i_hard = torch.cat((o_hard, x_hard), 1)
            i_cf_hard = torch.cat((o_cf_hard, x_cf_hard), 1)

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