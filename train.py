import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.jit import ScriptModule, script_method, trace
import time
import torch.nn.init as init
from cifar10 import load_data
from model import fractal_net
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
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class network(nn.Module):
    def __init__(self,deepest=False):
        super(network,self).__init__()
        self.dropout=[0., 0.1, 0.2, 0.3, 0.4]
        self.conv = [(3,64, 3, 3), (64,128, 3, 3), (128,256, 3, 3), (256,512, 3, 3), (512,512, 3, 3)]
        self.fractal_net=fractal_net(
        c=3, b=2, conv=self.conv,
        drop_path=0.15, dropout=self.dropout,
        deepest=deepest)
        self.linear=nn.Linear(8192,args.nb_classes)
        self.softmax=nn.Softmax(dim=1)
        init.kaiming_normal_(self.linear.weight)
        self.linear.bias.data.fill_(1)

    def forward(self,x):
        x=self.fractal_net(x)
        x=x.view(x.shape[0],-1)
        x=self.linear(x)
        x=self.softmax(x)
        return x
def train_network(data_loader,model,writer,criterion,optimizer,lr_scheduler):
    print("Training network")
    best_acc=.0
    model.train(mode=True)
    gl_losses = 0.0
    gl_prec1ss = 0.0
    gl_prec5ss = 0.0
    for i in range(args.nb_epochs):
        begin=time.time()
        losses = 0.0
        prec1ss = 0.0
        prec5ss = 0.0
        for j, (input, target) in enumerate(data_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
            input.requires_grad_()
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            losses=losses+float(loss.item())
            optimizer.step()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            prec1ss=prec1ss+float(prec1.item())
            prec5ss=prec5ss+float(prec5.item())
            be_time=time.time()-begin
            begin=time.time()
            if j!=0 and j % 100 == 0:
                print("train:epoch {} [{}/{}] loss is {:.3f},top1 is {:.3f}%,top5 is {:.3f}%,time is {:.3f}".format(i,
                                                                                                     j * args.batch_size,
                                                                                                     len(data_loader) * args.batch_size,
                                                                                                     float(losses/(100.0)),
                                                                                                     float(prec1ss / (100.0)),
                                                                                                     float(prec5ss / (100.0)),
                                                                                                     float(be_time)))
                gl_losses=gl_losses+losses/100.0
                gl_prec1ss=gl_prec1ss+prec1ss/100.0
                gl_prec5ss=gl_prec5ss+prec5ss/100.0
                losses = 0.0
                prec1ss = 0.0
                prec5ss = 0.0
        gl_losses = round(1.0 * gl_losses / float(len(data_loader)), 3)
        gl_prec1ss = round(1.0 * gl_prec1ss / float(len(data_loader)), 3)
        gl_prec5ss=round(1.0 * gl_prec5ss / float(len(data_loader)),3)
        if gl_prec1ss>best_acc:
            best_acc=gl_prec1ss
            torch.save(model.state_dict(), args.train_dir + "/model_best.pth")
        lr_scheduler.step()
        writer.add_scalar('loss', gl_losses, i)
        writer.add_scalar('loss', gl_prec1ss,i)
        writer.add_scalar('loss', gl_prec5ss,i)
        print("train:epoch {} loss is {:.3f},top1 is {:.3f}%,top5 is {:.3f}%".format(i,gl_losses,gl_prec1ss,gl_prec5ss))
        gl_prec1ss=0
        gl_prec5ss=0
        gl_losses=0
    writer.close()
    return model
def test_network(data_loader,model,criterion):
    losses=0.0
    prec1ss=0.0
    prec5ss=0.0
    model.eval()
    begin=time.time()
    with torch.no_grad():
        for j, (input, target) in enumerate(data_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda(non_blocking=True)
            input.requires_grad_()
            torch.cuda.synchronize()
            output = model(input)
            torch.cuda.synchronize()
            loss = criterion(output, target)
            losses = losses + float(loss.item())
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            prec1ss = prec1ss + float(prec1.item())
            prec5ss = prec5ss + float(prec5.item())
            be_time = time.time() - begin
            begin = time.time()
            if j!=0 and j % 100 == 0:
                print("train: [{}/{}] loss is {:.3f},top1 is {:.3f}%,top5 is {:.3f}%,time is {:.3f}".format(j * args.batch_size,
                                                                                             len(data_loader) * args.batch_size,
                                                                                             float(losses / (100.0)),
                                                                                             float(prec1ss / (100.0)),
                                                                                             float(prec5ss / (100.0)),
                                                                                             float(be_time)))
                losses = 0.0
                prec1ss = 0.0
                prec5ss = 0.0

if __name__=="__main__":
    torch.cuda.empty_cache()
    train_loader, test_loader = load_data(args.batch_size, args.batch_size)
    writer = SummaryWriter(args.train_dir)
    criterion = nn.CrossEntropyLoss()
    model=network(deepest=args.deepest)
    if torch.cuda.is_available():##将之放入GPU计算
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad, model.parameters()), args.learn_start,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    model=train_network(train_loader,model,writer,criterion,optimizer,lr_scheduler)
    test_network(test_loader,model,criterion)

