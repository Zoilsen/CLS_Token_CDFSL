import os
import random
import numpy as np
import torch
import torch.optim

from data.datamgr import SimpleDataManager, SetDataManager
from methods import backbone
from methods.backbone import model_dict
from methods.baselinetrain import BaselineTrain
from utils import load_state_to_the_backbone
from options import parse_args, get_resume_file, load_warmup_state

import time
from methods.protonet import ProtoNet
from methods.vision_transformer import VisionTransformer


def log(out, log_str):
    out.write(log_str + '\n')
    out.flush()
    print(log_str)



def train(base_loader, val_loader, model, start_epoch, stop_epoch, params, out, labeled_target_loader=None):
    # get optimizer and checkpoint path
    if params.stage == 'pretrain':
        if params.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), params.lr)
        elif params.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=0.9, nesterov=True, weight_decay=params.decay)
        elif params.optimizer == 'adamW':
            scratch_params = []
            pretrain_params = []
            halfPret_params = []
            for n, p in model.named_parameters():
                if 'feature' in n:
                    if 'domain_token' in n:
                        halfPret_params.append(p)
                    else:
                        pretrain_params.append(p)
                    
                    continue

                scratch_params.append(p)
            
            ratio = 0.0001
            optimizer = torch.optim.AdamW(
                    [{'params': pretrain_params, 'lr': params.lr * ratio}, 
                     {'params': halfPret_params, 'lr': params.lr * params.aux_param},
                     {'params': scratch_params}],
                    lr=params.lr,
                    weight_decay=params.decay)
            log(out, 'scale encoder lr by mul %s'%str(ratio)) 

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

    elif params.stage == 'metatrain': # not used
        if params.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), params.lr)
        elif params.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=0.9, nesterov=True, weight_decay=params.decay)
        elif params.optimizer == 'adamW':
            optimizer = torch.optim.AdamW(model.parameters(), params.lr, weight_decay=params.decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)


    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # for validation
    max_acc = {}
    params.eval_datasets.append('ave')
    for key in params.eval_datasets:
        max_acc[key] = -9999.0
    total_it = 0

    # start
    earliest_time = time.time()
    for epoch in range(start_epoch, stop_epoch):
        start_time = time.time()
        model.train()
        
        total_it = model.train_loop(epoch, base_loader, optimizer, total_it) #model are called by reference, no need to return
        model.eval()

        outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
        torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)       

        log_str = 'epoch: %d, '%epoch

        if ((epoch + 1) % params.eval_freq==0) or (epoch==stop_epoch-1):
            acc = model.test_loop(val_loader, params)
            for key in params.eval_datasets:
                if acc[key] > max_acc[key]:
                    max_acc[key] = acc[key]
                    outfile = os.path.join(params.checkpoint_dir, 'best_%s_model.tar'%key)
                    torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

                log_str = log_str + '%s: '%key[:4] + '{:.2f}, '.format(acc[key])

            log_str = log_str[:-2] + ' | max acc: '
            for key in params.eval_datasets:
                log_str = log_str + '%s: '%key[:4] + '{:.2f}, '.format(max_acc[key])
        
        log(out, log_str[:-2]) # the last two characters is a comma and a space

        if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
        scheduler.step()
        print('This epoch takes %d seconds' % (time.time() - start_time), 'still need around %.2f mins' % ((time.time() - start_time) * (stop_epoch - epoch) / 60))

    total_time = (time.time() - earliest_time) / 60
    print('Total time used %.2f mins' % total_time)

    return model


# --- main function ---
if __name__=='__main__':
    # output and tensorboard dir
    params = parse_args('train')
    params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
    params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    out = open(params.checkpoint_dir + '/log.txt', 'a')


    # set random seed
    seed = 0
    log(out, "set seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    log(out, '--- baseline training: {} ---\n'.format(params.name))
    log(out, str(params))


    # dataloader
    log(out, '\n--- prepare source dataloader ---')
    source_base_file = os.path.join(params.data_dir, 'miniImagenet', 'base.json')
    source_val_file = os.path.join(params.data_dir, 'miniImagenet', 'val.json')

    # model
    log(out, '\n--- build model ---')
    image_size = 224

    if params.stage == 'pretrain':
        log(out, '  pre-training the model using only the miniImagenet source data')
        base_datamgr = SimpleDataManager(image_size, batch_size=128)
        base_loader = base_datamgr.get_data_loader(source_base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(source_val_file, aug=False)

        model = BaselineTrain(model_dict[params.model], params.num_classes, params, tf_path=params.tf_dir)


    elif params.stage == 'metatrain':
        log(out, '  meta training the model using the miniImagenet data and the {} auxiliary data'.format(params.target_set))

        #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        n_query = 15 #max(1, int(16* params.test_n_way/params.train_n_way))

        train_few_shot_params = dict(n_way = params.train_n_way, n_support = params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(source_base_file, aug = params.train_aug )

        test_few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader( source_val_file, aug = False)

        ##########################
        model = ProtoNet(model_dict[params.model], params.train_n_way, params.n_shot)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()


    # load model
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume != '':
        resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            begin_epoch = tmp['epoch']+1

        #model.load_state_dict(tmp['state'])
        try:
            state = tmp['state']
        except KeyError:
            state = tmp['model_state']
        except:
            raise

        target_state_dict = model.state_dict()
        filtered_pretrained_state_dict = {k: v for k, v in state.items() if k in target_state_dict}
        model.load_state_dict(filtered_pretrained_state_dict, strict=False)

        log(out, '  resume the training with at {} epoch (model file {})'.format(begin_epoch, params.resume))

    # training
    log(out, '\n--- start the training ---')
    model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params, out)
