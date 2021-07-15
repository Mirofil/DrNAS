import os
import sys
sys.path.insert(0, './')
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from net2wider import configure_optimizer, configure_scheduler
from model_search import Network
from architect import Architect
import utils as utils
import wandb
import nasbench301 as nb
from genotypes import count_ops
from pathlib import Path
from tqdm import tqdm

from utils import genotype_depth, genotype_width
from copy import deepcopy

from sotl_utils import format_input_data, fo_grad_if_possible, hyper_meta_step, hypergrad_outer


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--k', type=int, default=6, help='init partial channel parameter')
#### regularization
parser.add_argument('--reg_type', type=str, default='l2', choices=['l2', 'kl'], help='regularization type')
parser.add_argument('--reg_scale', type=float, default=1e-3, help='scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2')

parser.add_argument('--higher_method' ,       type=str, choices=['val', 'sotl'],   default='sotl', help='Whether to take meta gradients with respect to SoTL or val set (which might be the same as training set if they were merged)')
parser.add_argument('--higher_params' ,       type=str, choices=['weights', 'arch'],   default='arch', help='Whether to do meta-gradients with respect to the meta-weights or architecture')
parser.add_argument('--higher_order' ,       type=str, choices=['first', 'second', None],   default="first", help='Whether to do meta-gradients with respect to the meta-weights or architecture')
parser.add_argument('--higher_loop' ,       type=str, choices=['bilevel', 'joint'],   default="bilevel", help='Whether to make a copy of network for the Higher rollout or not. If we do not copy, it will be as in joint training')
parser.add_argument('--higher_reduction' ,       type=str, choices=['mean', 'sum'],   default='sum', help='Reduction across inner steps - relevant for first-order approximation')
parser.add_argument('--higher_reduction_outer' ,       type=str, choices=['mean', 'sum'],   default='sum', help='Reduction across the meta-betach size')
parser.add_argument('--meta_algo' ,       type=str, choices=['reptile', 'metaprox', 'darts_higher', "gdas_higher", "setn_higher", "enas_higher"],   default=None, help='Whether to do meta-gradients with respect to the meta-weights or architecture')
parser.add_argument('--inner_steps', type=int, default=100, help='random seed')

args = parser.parse_args()

args.save = './experiments/{}/search-no_higher-progressive-{}-{}'.format(
    args.dataset, args.save, args.seed)
args.save += '-init_channels-' + str(args.init_channels)
args.save += '-layers-' + str(args.layers) 
args.save += '-init_pc-' + str(args.k)

try:
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
except Exception as e:
  print(f"Couldnt create exp dir due to {e}")

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
if args.dataset == 'cifar100':
    CIFAR_CLASSES = 100
def wandb_auth(fname: str = "nas_key.txt"):
  gdrive_path = "/content/drive/MyDrive/colab/wandb/nas_key.txt"
  if "WANDB_API_KEY" in os.environ:
      wandb_key = os.environ["WANDB_API_KEY"]
  elif os.path.exists(os.path.abspath("~" + os.sep + ".wandb" + os.sep + fname)):
      # This branch does not seem to work as expected on Paperspace - it gives '/storage/~/.wandb/nas_key.txt'
      print("Retrieving WANDB key from file")
      f = open("~" + os.sep + ".wandb" + os.sep + fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists("/root/.wandb/"+fname):
      print("Retrieving WANDB key from file")
      f = open("/root/.wandb/"+fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key

  elif os.path.exists(
      os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname
  ):
      print("Retrieving WANDB key from file")
      f = open(
          os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname,
          "r",
      )
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists(gdrive_path):
      print("Retrieving WANDB key from file")
      f = open(gdrive_path, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  wandb.login()
  
def load_nb301():
    version = '0.9'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
    model_paths_0_9 = {
        model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
    model_paths_1_0 = {
        model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

    # If the models are not available at the paths, automatically download
    # the models
    # Note: If you would like to provide your own model locations, comment this out
    if not all(os.path.exists(model) for model in model_paths.values()):
        nb.download_models(version=version, delete_zip=True,
                        download_dir=current_dir)

    # Load the performance surrogate model
    #NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
    #NOTE: Defaults to using the default model download path
    print("==> Loading performance surrogate model...")
    ensemble_dir_performance = model_paths['xgb']
    print(ensemble_dir_performance)
    performance_model = nb.load_ensemble(ensemble_dir_performance)
    
    return performance_model

    
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  run = wandb.init(project="NAS", group=f"Search_Cell_drNAS", reinit=True)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)


  import os
  from collections import namedtuple


  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, k=args.k,
                  reg_type=args.reg_type, reg_scale=args.reg_scale)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  if os.path.exists(Path(args.save) / "checkpoint.pt"):
    checkpoint = torch.load(Path(args.save) / "checkpoint.pt")
    # optimizer.load_state_dict(checkpoint["w_optimizer"])
    # architect.optimizer.load_state_dict(checkpoint["a_optimizer"])
    # scheduler.load_state_dict(checkpoint["w_scheduler"])
    
    model = checkpoint["model"].cuda()
    total_epoch = checkpoint["epoch"]
    all_logs = checkpoint["all_logs"]
    print(f"All_logs len= {len(all_logs)}")
    # alphas = checkpoint["alphas"]
    # for p1, p2 in zip(model._arch_parameters, alphas):
    #   p1.data = p2.data
    if total_epoch > 50:
      print(f"The training should already be over since we have epoch={total_epoch}!")
      start_epoch = total_epoch - 25
    else:
      start_epoch = total_epoch
    
  else:
    print(f"Path at {Path(args.save) / 'checkpoint.pt'} does not exist")
    start_epoch=0
    all_logs=[]

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.dataset=='cifar100':
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True)

  valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    pin_memory=True)

  architect = Architect(model, args)

  # configure progressive parameter
  epoch = 0
  ks = [6, 4]
  num_keeps = [7, 4]
  train_epochs = [2, 2] if 'debug' in args.save else [25, 25]
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min)

  api = load_nb301()
  
  if os.path.exists(Path(args.save) / "checkpoint.pt"):
    # checkpoint = torch.load(Path(args.save) / "checkpoint.pt")
    optimizer.load_state_dict(checkpoint["w_optimizer"])
    architect.optimizer.load_state_dict(checkpoint["a_optimizer"])
    scheduler.load_state_dict(checkpoint["w_scheduler"])
    epoch = start_epoch
    
  if start_epoch >= train_epochs[0]:
    print(f"Original start_epoch = {start_epoch}")
    epoch = start_epoch
    start_epoch = start_epoch - train_epochs[0]
    print(f"New start_epoch = {start_epoch}")
    train_epochs=train_epochs[1:]
    
    print(f"All logs len = {len(all_logs)}")


  if len(all_logs) >= sum(train_epochs):
    for log in all_logs:
      wandb.log(log)
    
  for i, current_epochs in tqdm(enumerate(train_epochs), desc = "Iterating over progressive phases"):
    for e in tqdm(range(start_epoch, current_epochs), desc = "Iterating over epochs"):
      if epoch >= 50:
        print(f"The trainign should be over since the total epoch={epoch}!")
        break
      lr = scheduler.get_lr()[0]
      logging.info('epoch %d lr %e', epoch, lr)

      genotype = model.genotype()
      logging.info('genotype = %s', genotype)
      model.show_arch_parameters()

      # training
      train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, e)
      logging.info('train_acc %f', train_acc)

      # validation
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
      
      log_epoch = epoch
      
      
      genotype_perf = api.predict(config=model.genotype(), representation='genotype', with_noise=False)
      ops_count = count_ops(genotype)
      width = {k:genotype_width(g) for k, g in [("normal", genotype.normal), ("reduce", genotype.reduce)]}
      depth = {k:genotype_depth(g) for k, g in [("normal", genotype.normal), ("reduce", genotype.reduce)]}

      logging.info(f"Genotype performance: {genotype_perf}, width: {width}, depth: {depth}, ops_count: {ops_count}")

      wandb_log = {"train_acc":train_acc, "train_loss":train_obj, "valid_acc":valid_acc, "valid_loss":valid_obj, 
                 "epoch":log_epoch, "search.final.cifar10":genotype_perf, "ops":ops_count, "alphas": model._arch_parameters, "width":width, "depth":depth}
      wandb.log(wandb_log)
      all_logs.append(wandb_log)
      
      epoch += 1
      scheduler.step()
      
      
      utils.save_checkpoint2({"model":model, "w_optimizer":optimizer.state_dict(), 
                           "a_optimizer":architect.optimizer.state_dict(), "w_scheduler":scheduler.state_dict(), "epoch": epoch, "all_logs":all_logs}, 
                          Path(args.save) / "checkpoint.pt")
      print(f"Saved checkpoint to {Path(args.save) / 'checkpoint.pt'}")
      # utils.save(model, os.path.join(args.save, 'weights.pt'))
    
    if not i == len(train_epochs) - 1:
      model.pruning(num_keeps[i+1])
      # architect.pruning([model.mask_normal, model.mask_reduce])
      model.wider(ks[i+1])
      optimizer = configure_optimizer(optimizer, torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay))
      scheduler = configure_scheduler(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min))
      logging.info('pruning finish, %d ops left per edge', num_keeps[i+1])
      logging.info('network wider finish, current pc parameter %d', ks[i+1])

  genotype = model.genotype()
  logging.info('genotype = %s', genotype)
  model.show_arch_parameters()
  
  logging.info(f"Printing all logs : {len(all_logs)}")

  for log in tqdm(all_logs, desc = "Logging all logs"):
    wandb.log(log)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)

    if epoch >= 10:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    optimizer.zero_grad()
    architect.optimizer.zero_grad()

    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    architect.optimizer.zero_grad()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)
    

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    if 'debug' in args.save:
      break

  return top1.avg, objs.avg

def train_higher(train_queue, valid_queue, network, architect, criterion, w_optimizer, a_optimizer, logger=None, 
                 inner_steps=100, epoch=0, steps_per_epoch=None, warm_start=10, args=None):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    train_iter = iter(train_queue)
    valid_iter = iter(valid_queue)
    search_loader_iter = zip(train_iter, valid_iter)
    for data_step, ((base_inputs, base_targets), (arch_inputs, arch_targets)) in tqdm(enumerate(search_loader_iter), total = round(len(train_queue)/inner_steps)):
      if steps_per_epoch is not None and data_step >= steps_per_epoch:
        break
      network.train()
      n = base_inputs.size(0)

      base_inputs = base_inputs.cuda()
      base_targets = base_targets.cuda(non_blocking=True)

      # get a random minibatch from the search queue with replacement
      input_search, target_search = next(iter(valid_queue))
      input_search = input_search.cuda()
      target_search = target_search.cuda(non_blocking=True)
      
      all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, 
                                                                                                search_loader_iter, inner_steps=inner_steps, epoch=epoch, args=args)

      network.zero_grad()

      model_init = deepcopy(network.state_dict())
      w_optim_init = deepcopy(w_optimizer.state_dict())
      arch_grads = [torch.zeros_like(p) for p in network.arch_parameters()]

      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
          if data_step in [0, 1] and inner_step < 3:
              print(f"Base targets in the inner loop at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
          logits = network(base_inputs)
          base_loss = criterion(logits, base_targets)
          base_loss.backward()
          w_optimizer.step()
          w_optimizer.zero_grad()
          if args.higher_method in ["val_multiple", "val"]:
              # if data_step < 2 and epoch < 1:
              #   print(f"Arch grads during unrolling from last step: {arch_grads}")
              logits = network(arch_inputs)
              arch_loss = criterion(logits, arch_targets)
              arch_loss.backward()
              with torch.no_grad():
                  for g1, g2 in zip(arch_grads, network.arch_parameters()):
                      g1.add_(g2)
              
              network.zero_grad()
              a_optimizer.zero_grad()
              w_optimizer.zero_grad()
              # if data_step < 2 and epoch < 1:
              #   print(f"Arch grads during unrolling: {arch_grads}")
              
      if args.higher_method in ["val_multiple", "val"]:
        print(f"Arch grads after unrolling: {arch_grads}")
        with torch.no_grad():
          for g, p in zip(arch_grads, network.arch_parameters()):
            p.grad = g
            

      
      if warm_start is None or (warm_start is not None and epoch >= warm_start):
        a_optimizer.step()
        a_optimizer.zero_grad()
        
        w_optimizer.zero_grad()
        architect.optimizer.zero_grad()
        # Restore original model state before unrolling and put in the new arch parameters
        new_arch = deepcopy(list(network.arch_parameters()))
        network.load_state_dict(model_init)
        
        with torch.no_grad():
            for p1, p2 in zip(network.arch_parameters(), new_arch):
                p1.data = p2.data
        
        for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
            if data_step in [0, 1] and inner_step < 3 and epoch % 5 == 0:
                logger.info(f"Doing weight training for real in higher_loop={args.higher_loop} at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
            logits = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            network.zero_grad()
            base_loss.backward()
            w_optimizer.step()
            n = base_inputs.size(0)

            prec1, prec5 = utils.accuracy(logits, base_targets, topk=(1, 5))

            objs.update(base_loss.item(), n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

        if data_step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', data_step, objs.avg, top1.avg, top5.avg)
        if 'debug' in args.save:
            break
      else:
        a_optimizer.zero_grad()
        w_optimizer.zero_grad()

    return  top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if step > 101:
        break
      input = input.cuda()
      target = target.cuda(non_blocking=True)
      
      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data, n)
      top1.update(prec1.data, n)
      top5.update(prec5.data, n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      if 'debug' in args.save:
        break

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 
