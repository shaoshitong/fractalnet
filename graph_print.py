import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.nn.init as init
from torch.jit import ScriptModule, script_method, trace
from torch.nn.parameter import Parameter
import argparse
from torch.autograd.variable import Variable
from graphviz import Digraph
from train import network
input_image=torch.rand(50,3,32,32)
"""
超参数配置
"""
parser = argparse.ArgumentParser(description='Weight Decay Experiments')
parser.add_argument('--nb_classes', dest='nb_classes', help='the number of classes', default=10, type=int)
parser.add_argument('--nb_epochs', dest='nb_epochs', help='the number of epochs', default=200, type=int)
parser.add_argument('--learn_start', dest='learn_start', help='the learning rate at begin', default=0.02, type=float)
parser.add_argument('--batch_size', dest='batch_size', help='training batch size', default=64, type=int)
parser.add_argument('--momentum', dest='momentum', help='the momentum', default=0.5, type=float)
parser.add_argument('--schedule', dest='schedule', help='weight Decrease learning rate',default=[100,150],type=int,nargs='+')
parser.add_argument('--gamma', dest='gamma', help='gamma', default=0.1, type=float)
parser.add_argument('--train_dir', dest='train_dir', help='training data dir', default="tmp", type=str)
parser.add_argument('--deepest', dest='deepest',help='Build with only deepest column activated',default=True,type=bool)
parser.add_argument('--weight_decay', dest='weight_decay', help='weight decay', default=1e-4, type=float)
parser.add_argument('--load', dest='load',help='Test network with weights file',default=True,type=bool)
parser.add_argument('--test-all', nargs=1,help='Test all the weights from a folder')
parser.add_argument('--summary',help='Print a summary of the network and exit',action='store_true')
args = parser.parse_args()

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot

if __name__=="__main__":
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss().cuda()
    model = network(deepest=args.deepest).cuda()
    optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad, model.parameters()), args.learn_start,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    x = Variable(torch.randn(50, 3, 32, 32)).cuda()
    y=model(x)
    g=make_dot(y)
    g.view()