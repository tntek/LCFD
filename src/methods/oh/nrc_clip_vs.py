import argparse
import os, sys
sys.path.append('./')

import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.models import network
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
import pickle
from src.utils.utils import *
from torch import autograd
from clip.custom_clip import get_coop
from src.utils import IID_losses,miro,loss
import clip
from data.imagnet_prompts import imagenet_classes
import time
from src.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class ImageList_idx(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9*dsize)
    # print(dsize, tr_size, dsize - tr_size)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs * 3,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders

def test_time_tuning(model, inputs, optimizer, args,target_output,im_re_o):

    target_output = target_output.cuda()

    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():

            output_logits = model(inputs)             
            if(output_logits.shape[0]!=args.batch_size):
                padding_f=torch.zeros([args.batch_size-output_logits.shape[0],output_logits.shape[1]],dtype=torch.float).cuda()
                output_logits = torch.cat((output_logits, padding_f.float()), 0)
                target_output =  torch.cat((target_output, padding_f.float()), 0)

            im_loss_o, Delta = im_re_o.update(output_logits,target_output)
            Delta = 1.0/(Delta+1e-5)
            Delta = nn.Softmax(dim=1)(Delta)
            output_logits_sm = nn.Softmax(dim=1)(output_logits)
            output = Delta*output_logits_sm
            iic_loss = IID_losses.IID_loss(output, output_logits_sm)
            loss = 0.5*(iic_loss - 0.0003*im_loss_o) #0.0003
            # loss = 0.5*(iic_loss)

            if(inputs.shape[0]!=args.batch_size):
                output = output[:inputs.shape[0]]
                target_output = target_output[:inputs.shape[0]]



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return output,loss


def test_time_adapt_eval(input,target, model, optimizer, optim_state, args,target_output,im_re_o):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    start_test = True
    progress = ProgressMeter(
        input.shape[0],
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.train()
            im_re_o.train()
    end = time.time()

        
    if not args.cocoop: # no need to reset cocoop because it's fixed
        if args.tta_steps > 0:
            with torch.no_grad():
                model.train()
                im_re_o.train()
        optimizer.load_state_dict(optim_state)
        output,loss_ib = test_time_tuning(model, input, optimizer, args,target_output,im_re_o)

    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model.eval()
            im_re_o.eval()
            output= model(input)
    output = output.cpu()
    # acc1, acc5 = accuracy(output, target, topk=(1, 5))
    # print("acc1:")
    # print(acc1)
  
    # top1.update(acc1[0],input.shape[0])
    # top5.update(acc5[0],input.shape[0])

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # if (i+1) % args.print_freq == 0:
    #     progress.display(i)
    # i = i+1
    # iter_num = iter_num+1
    # progress.display_summary()

    return output,loss_ib


def clip_pre_text(args):
    List_rd = []
    if 'image' in args.dset:
        # classnames = imagenet_classes
        classnames_all = imagenet_classes
        classnames = []
        if args.dset.split('_')[-1] in ['a','r','v']:
            label_mask = eval("imagenet_{}_mask".format(args.dset.split('_')[-1]))
            if 'r' in args.dset:
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    else:
        with open(args.name_file) as f:
            for line in f:
                List_rd.extend([i for i in line.split()])
        f.close()
        classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    args.classname = classnames
    prompt_prefix = args.ctx_init.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts


def train_target(args):
    text_inputs = clip_pre_text(args)
    DOWNLOAD_ROOT = '~/.cache/clip'
    device = int(args.gpu_id)
    clip_model,_,_ = clip.load(args.arch,device=device, download_root=DOWNLOAD_ROOT)
    clip_model.float()
    clip_model.eval()

    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))


    target_logits = torch.ones(args.batch_size,args.class_num)
    im_re_o = miro.MIRO(target_logits.shape).cuda()
    del target_logits
    model = get_coop(args.arch, args.dset, args.gpu, args.n_ctx, args.ctx_init)

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        #if k.find('bn')!=-1:
        if True:
            param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in netB.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': args.lr * 1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    param_group_ib = []
    for k, v in im_re_o.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': args.lr * args.lr_decay3}]

    for k, v in model.prompt_learner.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': args.lr * args.lr_decay3}]
    optimizer_ib = optim.SGD(param_group_ib)
    optimizer_ib = op_copy(optimizer_ib)
    optim_state = deepcopy(optimizer_ib.state_dict())
    classnames = args.classname
    model.reset_classnames(classnames, args.arch)
    acc_init = 0


    start = True
    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,512)
    score_bank = torch.randn(num_sample, 12).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    model.eval()
    im_re_o.eval()

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm=F.normalize(output)
            outputs = netC(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log=0
    while iter_num < max_iter:
        
        # comment this if on office-31
        # if iter_num>0.5*max_iter:
        #     args.K = 5
        #     args.KK = 4

        #for epoch in range(args.max_epoch):


        #iter_target = iter(dset_loaders["target"])
        try:
            inputs_test, labels, tar_idx = next(iter_target)
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, labels, tar_idx = next(iter_target)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        # uncomment this if on office-31
        #lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        inputs_target = inputs_test.cuda()

        with torch.no_grad():
            outputs_test_new = netC(netB(netF(inputs_test))).detach()
        netF.eval()
        netB.eval()
        netC.eval()
        model.train()
        im_re_o.train()

        output_clip,_ = test_time_adapt_eval(inputs_test,labels, model, optimizer_ib, optim_state,args,outputs_test_new,im_re_o)
        # with torch.no_grad():
        output_clip = output_clip.detach().cuda().float()
        output_clip_sm = nn.Softmax(dim=1)(output_clip)
        netF.train()
        netB.train()
        netC.train()
        model.eval()
        im_re_o.eval()

        features_test = netF(inputs_target)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=args.K+1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]    #batch x K x C

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
            _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=args.KK+1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:,:,1:] # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (
                idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            #weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                               args.class_num)  # batch x KM x C

            score_self = score_bank[tar_idx]

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK,
                                                    -1)  # batch x C x 1
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        loss = torch.mean(const)

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,
                                                         -1)  # batch x K x C

        loss += torch.mean((
            F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
            weight.cuda()).sum(1))

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax *
                                    torch.log(msoftmax + args.epsilon))
        loss += gentropy_loss

        if (args.loss_func=="l1"):
            # print("using l1 loss")
            loss_l1 = torch.nn.L1Loss(reduction='mean')
            classifier_loss = loss_l1(softmax_out, output_clip_sm)
            classifier_loss *= args.cls_par
        elif (args.loss_func=="l2"):
            loss_l2 = torch.nn.MSELoss(reduction='mean')
            classifier_loss = loss_l2(softmax_out,output_clip_sm)
            classifier_loss *= args.cls_par
        elif (args.loss_func=="iid"):
            classifier_loss = IID_losses.IID_loss(softmax_out,output_clip_sm)
            classifier_loss *= args.cls_par
        elif (args.loss_func=="kl"):
            classifier_loss = F.kl_div(softmax_out.log(),output_clip_sm, reduction='sum')
            classifier_loss *= args.cls_par
        elif (args.loss_func=="sce"):
            _, pred = torch.max(output_clip, 1)
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else :
            classifier_loss = torch.tensor(0.0).cuda()


        loss = loss + classifier_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc_vs(dset_loaders['test'], netF, netB,
                                             netC,flag= True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
                    args.name, iter_num, max_iter, acc_s_te
                ) + '\n' + 'T: ' + acc_list
            else:
                acc_s_te, _ = cal_acc_vs(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

            if acc_s_te>acc_log:
                acc_log=acc_s_te
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + '2021_'+str(args.K) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.K) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.K) + ".pt"))

    return netF, netB, netC
