import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
import clip
from src.models import network,shot_model
from src.utils import loss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from numpy import dtype, float32, linalg as LA
from src.data.datasets.data_loading import get_test_loader
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK,IMAGENET_V_MASK
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from src.models.model import *
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

DOWNLOAD_ROOT = '~/.cache/clip'
device = int(0)
clip_model,_,_ = clip.load('ViT-B/32',device=device, download_root=DOWNLOAD_ROOT)
clip_model.float()
clip_model.eval()
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(args):
    text_inputs = clip_pre_text(args)
    if 'image' in args.dset:
        if args.net[0:3] == 'res':
            netF = network.ResBase(res_name=args.net)
        elif args.net[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=args.net)
        netC = network.Net2(2048,1000)
        base_model,transform = get_model(args, args.class_num)
        netC.linear.load_state_dict(base_model.fc.state_dict())
        del base_model
        Shot_model = shot_model.OfficeHome_Shot(netF,netC)
        # base_model = normalize_model(Shot_model, transform.mean, transform.std)
        base_model = Shot_model
        if args.dset == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif args.dset == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif args.dset == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)
        elif args.dset == "imagenet_v":
            base_model = ImageNetXWrapper(base_model, IMAGENET_V_MASK)
    else :
        base_model = get_model(args, args.class_num)

    base_model.cuda()
    
    # for k, v in netC.named_parameters():
    #     if args.lr_decay1 > 0:
    #         param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
    #     else:
    #         v.requires_grad = False

    param_group = []
    for k, v in base_model.named_parameters():
        if 'netC' in k or 'fc' in k:
            v.requires_grad = False
        else:
            param_group += [{'params': v, 'lr': args.lr}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    args.SEVERITY = [5]
    args.ADAPTATION = 'tent'
    args.NUM_EX = -1
    args.ALPHA_DIRICHLET = 0.0
    # dom_names_loop = ["mixed"] if "mixed_domains" in args.SETTING else dom_names_all
    domain_name = args.type[args.t]
    dom_names_all = args.type
    target_data_loader = get_test_loader(setting=args.SETTING,
                                        adaptation=args.ADAPTATION,
                                        dataset_name=args.dset,
                                        root_dir=args.data,
                                        domain_name=domain_name,
                                        severity=args.level,
                                        num_examples=args.NUM_EX,
                                        rng_seed=args.seed,
                                        domain_names_all=dom_names_all,
                                        alpha_dirichlet=args.ALPHA_DIRICHLET,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        workers=args.worker)
    max_iter = args.max_epoch * len(target_data_loader)
    interval_iter = max_iter // args.interval
    iter_num = 0

    test_data_loader = get_test_loader(setting=args.SETTING,
                                    adaptation=args.ADAPTATION,
                                    dataset_name=args.dset,
                                    root_dir=args.data,
                                    domain_name=domain_name,
                                    severity=args.level,
                                    num_examples=args.NUM_EX,
                                    rng_seed=args.seed,
                                    domain_names_all=dom_names_all,
                                    alpha_dirichlet=args.ALPHA_DIRICHLET,
                                    batch_size=args.batch_size*3,
                                    shuffle=False,
                                    workers=args.worker)


    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, _, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            base_model.eval()
            mem_label = obtain_label(test_data_loader, base_model, args,text_inputs)
            #mem_label = torch.from_numpy(mem_label).cuda()
            mem_label = torch.from_numpy(mem_label).cuda()
            base_model.train()

        inputs_test = inputs_test.cuda()
        # inputs_test_augs = inputs_test_augs[0].cuda() 
        # inputs_test = inputs_test_augs[0].cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        outputs_test = base_model(inputs_test)        
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()


        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        #creation=nn.MSELoss()
        #loss=creation(softmax_out,clip_score)
        #classifier_loss = classifier_loss + 1.0 * loss

        # msoftmax = softmax_out.mean(dim=0)
        # classifier_loss = classifier_loss + 2.0 * loss_info

        # if  args.dset=='office':
        #     gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
        #     classifier_loss = classifier_loss - 1.1 * gentropy_loss
        # if  args.dset=='office-home':
        #     gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
        #     classifier_loss = classifier_loss - 0.6 * gentropy_loss
        # if  args.dset=='VISDA-C':
        #     gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
        #     classifier_loss = classifier_loss - 0.1 * gentropy_loss
        
        # """
        # if args.ent:
        #     softmax_out = nn.Softmax(dim=1)(outputs_test)
        #     entropy_loss = torch.mean(loss.Entropy(softmax_out))
        #     if args.gent:
        #         msoftmax = softmax_out.mean(dim=0)
        #         gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
        #         entropy_loss -= gentropy_loss
        #     im_loss = entropy_loss * args.ent_par
        #     classifier_loss += im_loss
        # """

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(test_data_loader, base_model, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(args.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(test_data_loader, base_model, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(args.name, iter_num, max_iter, acc_s_te,classifier_loss)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            base_model.train()
        

    if args.issave:   
        torch.save(base_model.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))

        
    return base_model


def print_args(args):
    s = "==========================================\n"    
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, base_model, args,text_inputs):

    m = nn.BatchNorm1d(512).cuda()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs_cuda = inputs.cuda()
            clip_feature, clip_score = clip_pre(text_inputs,inputs_cuda)
            clip_score = torch.tensor(clip_score).cpu()
            inputs = inputs.cuda() 
            if 'image' in args.dset:
                # feas = base_model.model.netF(base_model.normalize(inputs))
                feas = base_model.netF(inputs)
                if 'k' in args.dset:
                    outputs = base_model.netC(feas)
                else:
                    outputs = base_model.masking_layer(base_model.netC(feas))
                # outputs = base_model(inputs)
            else:
                feas = base_model.encoder(inputs)
                outputs = base_model.fc(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                clip_all_output = clip_score.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                clip_all_output = torch.cat((clip_all_output, clip_score.float().cpu()), 0)
                _, predict = torch.max(all_output, 1)
                #_, predict_1 = torch.max(clip_all_output, 1)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    all_output = nn.Softmax(dim=1)(all_output)
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)
    _, clip_predict = torch.max(clip_all_output, 1)


    clip_predict = clip_predict.int().numpy()

    acc = np.sum(clip_predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}'.format(acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    
    #mix_soft = mix_soft.cuda()
    #zero = torch.zeros_like(mix_soft)
    #one = torch.ones_like(mix_soft)
    #mix_soft_dis = torch.where(mix_soft>0.4,one,zero)
    #mix_soft_dis = torch.where(mix_soft>0.4,1,0)
    return clip_predict.astype('int')



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


def clip_text(text_inputs,image_features):
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #_, top_labels = similarity.cpu().topk(1,dim=-1)
    #_, predict = torch.max(similarity, 1)
    return similarity,text_features

def clip_pre(text_inputs,inputs_test):
    with torch.no_grad():
        image_features = clip_model.encode_image(inputs_test)
        logits_per_image, logits_per_text = clip_model(inputs_test, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return image_features,probs

def clip_image(inputs_test):
    image_input = inputs_test.cuda()
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features
