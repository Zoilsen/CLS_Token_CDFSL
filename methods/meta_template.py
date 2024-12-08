import torch.nn as nn
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter



class MetaTemplate(nn.Module):
  def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
    super(MetaTemplate, self).__init__()
    self.n_way      = n_way
    self.n_support  = n_support
    self.n_query    = -1 #(change depends on input)
    self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
    self.feat_dim   = self.feature.final_feat_dim
    self.change_way = change_way  #some methods allow different_way classification during training and test
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

  @abstractmethod
  def set_forward(self, x, is_feature):
    pass

  @abstractmethod
  def set_forward_loss(self, x):
    pass


  def forward(self, x):
    out = self.feature.forward(x)
    return out

  def parse_feature(self, x, is_feature, params):
    x = x.cuda()
    if is_feature:
      z_all = x
    else:
      x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
      z_all = self.feature.forward(x, params=params)
      z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
     
    z_support = z_all[:, :self.n_support]
    z_query = z_all[:, self.n_support:]

    return z_support, z_query


  def correct(self, x):
    scores, loss = self.set_forward_loss_for_test(x)

    y_query = np.repeat(range( self.n_way ), self.n_query )
    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    return float(top1_correct), len(y_query), loss.item()*len(y_query)


  def train_loop(self, epoch, train_loader, optimizer, total_it):
    avg_loss=0
    for i, (x,_) in enumerate(train_loader):
        #print(x.size()) [5,20,3,224,224]
        self.n_query = x.size(1) - self.n_support
        if self.change_way:
            self.n_way  = x.size(0)
        optimizer.zero_grad()
        loss = self.set_forward_loss(x)[1] # return [scores, loss]
        loss.backward()
        optimizer.step()
        avg_loss = avg_loss+loss.item()
        total_it += 1

    return total_it


  def test_loop(self, test_loader, params):
    import network_test
    params.ckp_path = params.checkpoint_dir + '/last_model.tar'
    train_dataset = params.dataset
    n_shot = params.n_shot
    params.n_shot = 5
    acc_dict = {}
    novel_accs = []
    for d in params.eval_datasets:
        #print('test on', d)
        if d == 'ave':
            continue
        
        params.dataset = d
        output = network_test.test_single_ckp(params)
        acc = float(output.split('Acc = ')[-1].split('%')[0])
        acc_dict[d] = acc

        if d != 'miniImagenet':
            novel_accs.append(acc)

    acc_dict['ave'] = sum(novel_accs) / len(novel_accs)

    params.dataset = train_dataset
    params.n_shot = n_shot

    return acc_dict
