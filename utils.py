import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import defaultdict

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
  def __init__(self, length):
    self.length = length

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img


def _data_transforms_svhn(args):
  SVHN_MEAN = [0.4377, 0.4438, 0.4728]
  SVHN_STD = [0.1980, 0.2010, 0.1970]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length,
                                             args.cutout_prob))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  return train_transform, valid_transform


def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
  CIFAR_STD = [0.2673, 0.2564, 0.2762]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length,
                                             args.cutout_prob))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  return train_transform, valid_transform


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def process_step_vector(x, method, mask, tau=None):
  if method == 'softmax':
    output = F.softmax(x, dim=-1)
  elif method == 'dirichlet':
    output = torch.distributions.dirichlet.Dirichlet(
      F.elu(x) + 1).rsample()
  elif method == 'gumbel':
    output = F.gumbel_softmax(x, tau=tau, hard=False, dim=-1)
  
  if mask is None:
    return output
  else:
    output_pruned = torch.zeros_like(output)
    output_pruned[mask] = output[mask]
    output_pruned /= output_pruned.sum()
    assert (output_pruned[~mask] == 0.0).all()
    return output_pruned
    

def process_step_matrix(x, method, mask, tau=None):
  weights = []
  if mask is None:
    for line in x:
      weights.append(process_step_vector(line, method, None, tau))
  else:
    for i, line in enumerate(x):
      weights.append(process_step_vector(line, method, mask[i], tau))
  return torch.stack(weights)


def prune(x, num_keep, mask, reset=False):
  if not mask is None:
    x.data[~mask] -= 1000000
  src, index = x.topk(k=num_keep, dim=-1)
  if not reset:
    x.data.copy_(torch.zeros_like(x).scatter(dim=1, index=index, src=src))
  else:
    x.data.copy_(torch.zeros_like(x).scatter(dim=1, index=index, src=1e-3*torch.randn_like(src)))
  mask = torch.zeros_like(x, dtype=torch.bool).scatter(
      dim=1, index=index, src=torch.ones_like(src,dtype=torch.bool))
  return mask

def save_checkpoint2(state, filename):
  from os import path as osp
  if osp.isfile(filename):
    os.remove(filename)
  try:
    torch.save(state, filename.parent / (filename.name + 'tmp'))
    os.replace(filename.parent / (filename.name + 'tmp'), filename)
  except Exception as e:
    print(f"Failed to save new checkpoint into {filename} due to {e}")
  assert osp.isfile(filename), 'save filename : {:} failed, which is not found.'.format(filename)
  return filename



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def genotype_to_adjacency_list(genotype, steps=4):
  # Should pass in genotype.normal or genotype.reduce
  G = defaultdict(list)
  for nth_node, connections in enumerate(chunks(genotype, 2), start=2): # Darts always keeps two connections per node and first two nodes are fixed input
    for connection in connections:
      G[connection[1]].append(nth_node)
  # Add connections from all intermediate nodes to Output node
  for intermediate_node in [2,3,4,5]:
    G[intermediate_node].append(6)
  return G
    
def DFS(G,v,seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths
  
def count_edges_along_path(genotype, path):
  count = 0
  for i in range(1, len(path)-1): #Leave out the first and last nodes
    idx_in_genotype = path[i]-2
    relevant_edges = genotype[idx_in_genotype*2:idx_in_genotype*2+2]
    for edge in relevant_edges:
      if edge[1] == path[i-1]:
        count += 1
  return count

def genotype_depth(genotype):
  # The shortest path can start in either of the two input nodes
  all_paths0 = DFS(genotype_to_adjacency_list(genotype), 0)
  all_paths1 = DFS(genotype_to_adjacency_list(genotype), 1)

  cand0 = max(len(p)-1 for p in all_paths0)
  cand1 = max(len(p)-1 for p in all_paths1)
  
  # max_paths0 = [p for p in all_paths0 if len(p) == cand0]
  # max_paths1 = [p for p in all_paths1 if len(p) == cand1]

  # path_depth0 = max([count_edges_along_path(genotype, p) for p in max_paths0])
  # path_depth1 = max([count_edges_along_path(genotype, p) for p in max_paths1])
  
  # return max(path_depth0, path_depth1)

  return max(cand0, cand1)

def genotype_width(genotype):
  width = 0
  for edge in genotype:
    if edge[1] in [0, 1]:
      width += 1/2
  return width