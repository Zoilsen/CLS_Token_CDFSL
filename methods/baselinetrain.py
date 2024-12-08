import torch.nn as nn
from tensorboardX import SummaryWriter

from methods import backbone
import torch.nn.functional as F
import torch
from methods.protonet import ProtoNet

import network_test



# --- conventional supervised training ---
class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, params, tf_path=None, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        
        self.params = params
        # feature encoder
        self.feature = model_func()
      
        if 'VIT' in params.model:
            self.feature.final_feat_dim = self.feature.num_features
        else:
            self.feature.final_feat_dim = 512

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class, bias=False)

        elif loss_type == 'dist':
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type
        self.loss_fn = nn.CrossEntropyLoss()

        self.num_class = num_class

        ### load kmeans cluster ###
        inp = open('kmeans_cluster=%d.txt'%(self.feature.num_domain_token), 'r')
        label_dict = {}
        cluster_dict = {}
        for l in inp.readlines():
            cluster_id, cls_ids = l[:-2].split(':')
            cls_ids = cls_ids.split(',')
            cluster_dict[cluster_id] = [int(cls) for cls in cls_ids]
            for cls in cls_ids:
                label_dict[int(cls)] = int(cluster_id)
        inp.close()
        self.label_dict = label_dict
        self.cluster_dict = cluster_dict


    def forward_loss(self, x, y, epoch):
        self.params.aux_container['epoch'] = epoch
        # forward feature extractor
        x = x.cuda()
        y = y.cuda(); raw_y = y
        y = y % self.num_class
       
        ### add domain_token to input cls_token ###
        domain_tokens = self.feature.domain_token
        num_dm = len(self.cluster_dict.keys()); assert num_dm == self.feature.num_domain_token
        dm_y = torch.tensor([self.label_dict[int(label)] for label in y]).to(torch.int64).cuda()
        token_mask = F.one_hot(dm_y, num_dm) # [b, n_dm]
        token_mask = token_mask.unsqueeze(-1) # [b, n_dm, 1]
        domain_token = (token_mask * domain_tokens).sum(dim=1, keepdim=True) # [b, 1, c]

        cls_token = domain_token + self.feature.cls_token.detach()
               
        x_map = self.feature.forward(x, params=self.params, input_cls_token=cls_token) # [b, num_token, c]
        ###########################################

        x = x_map[:, 0] # cls token, [b, c]
        CLS_scores = self.classifier.forward(x)        
        
        # calculate loss
        loss_CLS = self.loss_fn(CLS_scores, y)

        if self.params.clsToken_domainToken_orth_w != 0.0:
            cls_token = self.feature.cls_token # [1, 1, c]
            cls_token = F.normalize(cls_token.squeeze(0), dim=-1) # [1, c]
            domain_token = self.feature.domain_token # [1, nd, c]
            domain_token = F.normalize(domain_token.squeeze(0), dim=-1) # [nd, c]
            clsToken_domainToken_orth_loss = torch.mean(torch.matmul(cls_token, domain_token.t()).abs()) # [1, nd] -> []

            loss_CLS = loss_CLS + self.params.clsToken_domainToken_orth_w * clsToken_domainToken_orth_loss


        return loss_CLS


    def train_loop(self, epoch, train_loader, optimizer, total_it):
        print_freq = len(train_loader) // 10
        avg_loss=0
        
        self.params.aux_container['is_new_epoch'] = True

        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            self.params.aux_container['batch_id'] = i
            loss = self.forward_loss(x, y, epoch)#, name)
            self.params.aux_container['is_new_epoch'] = False
            
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()

            #if (i + 1) % print_freq==0:
            #    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)))
            #if (total_it + 1) % 10 == 0:
            #    self.tf_writer.add_scalar('loss', loss.item(), total_it + 1)
            total_it += 1

        return total_it

    def test_loop(self, val_loader, params):       
        params.ckp_path = params.checkpoint_dir + '/last_model.tar'
        train_dataset = params.dataset
        acc_dict = {}
        novel_accs = []
        for d in params.eval_datasets:
            if d == 'ave':
                continue

            params.dataset = d
            output = network_test.test_single_ckp(params)
            acc = float(output.split('Acc = ')[-1].split('%')[0])
            acc_dict[d] = acc

            if d != 'miniImagenet':
                novel_accs.append(acc)

        if len(novel_accs) == 0:
            acc_dict['ave'] = sum(novel_accs) # [0]
        else:
            acc_dict['ave'] = sum(novel_accs) / len(novel_accs)

        params.dataset = train_dataset
        
        return acc_dict





















