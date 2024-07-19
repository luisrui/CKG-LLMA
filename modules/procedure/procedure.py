from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import torch
from ..model import KLMCR
from ..utils import Contrast

def TransR_train(args, kg_train_data, Recmodel, opt, device):
    Recmodel.train()
    kgloader = DataLoader(kg_train_data, batch_size=args['kge_batch_size'], drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(device)
        relations = data[1].to(device)
        pos_tails = data[2].to(device)
        neg_tails = data[3].to(device)
        kg_batch_loss = Recmodel.calc_kg_loss_transE(
            heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()

def BPR_train_contrast(
        args, 
        rec_data, 
        Recmodel : KLMCR, 
        contrast_model : Contrast, 
        contrast_views, 
        optimizer):
    Recmodel.train()
    batch_size = args['bpr_batch_size']
    device = args['device']
    dataloader = DataLoader(rec_data, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=12)

    total_batch = len(dataloader)
    total_loss = 0.
    bpr_loss = 0.
    con_loss = 0.
    # For SGL
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]
    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=True):
        batch_users = train_data[0].long().to(device)
        batch_pos = train_data[1].long().to(device)
        batch_neg = train_data[2].long().to(device)

        # main task (batch based)
        # bpr loss for a batch of users
        l_bpr_reg, l_bpr = Recmodel.calc_bpr_loss(batch_users, batch_pos, batch_neg)
        l_ssl = list()
        items = batch_pos  # [B*1]

        usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, kgv1)
        usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, kgv2)

        # from SGL source
        items_uiv1 = itemsv1_ro[items - rec_data.num_users]
        items_uiv2 = itemsv2_ro[items - rec_data.num_users]
        l_item = contrast_model.info_nce_loss_overall(
            items_uiv1, items_uiv2, itemsv2_ro)

        users = batch_users
        users_uiv1 = usersv1_ro[users]
        users_uiv2 = usersv2_ro[users]
        l_user = contrast_model.info_nce_loss_overall(
            users_uiv1, users_uiv2, usersv2_ro)
        # l_user = contrast_model.grace_loss(users_uiv1, users_uiv2)
        # L = l_bpr_reg + L_user + L_item + L_kg + R^2
        l_ssl.extend([l_user*args['ssl_reg'], l_item*args['ssl_reg']])

        if l_ssl:
            l_ssl = torch.stack(l_ssl).sum()
            l_all = l_bpr_reg+l_ssl
            con_loss += l_ssl.cpu().item()
        else:
            l_all = l_bpr_reg

        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        bpr_loss += l_bpr_reg.cpu().item()
        total_loss += l_all.cpu().item()
        # if world.tensorboard:
        #     w.add_scalar(f'BPRLoss/BPR', l_all, epoch *
        #                  int(len(users) / batch_size) + batch_i)
    total_loss = total_loss / (total_batch*batch_size)
    bpr_loss = bpr_loss / (total_batch*batch_size)
    con_loss = con_loss / (total_batch*batch_size)

    return total_loss, bpr_loss, con_loss
