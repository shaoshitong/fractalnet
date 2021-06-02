import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.jit import ScriptModule, script_method, trace
from torch.nn.parameter import Parameter
class myenumerate:
    def __init__(self, wrapped,end=None):
        self.wrapped = wrapped
        self.offset = 0
        if end == None:
            self.end = len(wrapped)
        else:
            self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= len(self.wrapped) or self.offset>=self.end:
            raise (StopIteration)
        else:
            item = self.wrapped[self.offset]
            self.offset += 1
            return self.offset-1,item
def pytorch_categorical(count, seed):
    global gts
    assert count > 0
    arr = torch.tensor([1.] + [.0 ]*(count-1),dtype=torch.float64)
    if seed is not None:
       gts=torch.manual_seed(seed)
    index=torch.randperm(count,dtype=torch.long,generator=gts)
    # seed = None 则每次顺序都是随机的
    return torch.index_select(arr,0,index)
def rand_one_in_array(count, seed=None):
    if seed is None:
        seed = np.random.randint(1, high=int(1e6))
    return pytorch_categorical(count=count, seed=seed)
class mulLayer(nn.Module):
    def __init__(self, global_path):
        super(mulLayer, self).__init__()
        if torch.is_tensor(global_path)==False:
            self.global_path=Parameter(torch.tensor(global_path,dtype=torch.float64))
        else:
            self.global_path = Parameter(global_path.float())

    def forward(self,x):
        while self.global_path.ndim !=x.ndim:
            self.global_path=Parameter(self.global_path.unsqueeze(-1))
        x=torch.sum(x* self.global_path[:x.shape[0]], dim=0, keepdim=True)
        return x
class JoinLayer(nn.Module):
    def __init__(self, drop_p, is_global, global_path, force_path):
        super(JoinLayer, self).__init__()
        self.p = 1. - drop_p
        self.is_global = is_global
        self.global_path=mulLayer(global_path)
        self.force_path = force_path
        self.average_shape=None
    def _weights_init_list(self,x):
        self.average_shape = x.shape.tolist[1:]
    def _random_arr(self, count, p):
        return torch.distributions.Binomial(1, torch.tensor([p],dtype=torch.float64).expand(count)).sample()
    def _arr_with_one(self, count):
        return rand_one_in_array(count=count)
    def _gen_local_drops(self, count, p):
        arr = self._random_arr(count, p)
        if torch.sum(arr).item()==0:
            return self._arr_with_one(count)
        else:
            return arr
    def _gen_global_path(self, count):
        return self.global_path[:count]

    def _drop_path(self, inputs):
        if self.average_shape is None:
            self.average_shape=inputs.shape[1:]
        count = inputs.shape[0]
        if self.is_global == True:
            drops=self.global_path.global_path[:inputs.shape[0]]
            ave = self.global_path(inputs)
        else:
            drops = self._gen_local_drops(count, self.p)
            while drops.ndim != inputs.ndim:
                drops = drops.unsqueeze(-1)
            ave = torch.sum(inputs * drops[:inputs.shape[0]], dim=0, keepdim=True)
        indexsum = torch.sum(drops).item()
        return ave.squeeze(0)/indexsum if indexsum else ave.squeeze(0)
    def _ave(self, inputs):
        return torch.mean(inputs.float(), dim=0, keepdim=True)

    def forward(self,inputs):
        inputs= self._drop_path(inputs) if self.force_path or inputs.requires_grad else self._ave(inputs)
        return inputs
"""
a=torch.autograd.Variable(torch.tensor([[[5,3,1],[4,3,2],[3,3,3]],[[5,3,1],[4,6,2],[3,4,3]]],dtype=torch.float64),requires_grad=True)
b=JoinLayer(0.3,False,[2,2,4,5],True)
c=b(a)
torch.sum(c,dtype=torch.float64).backward()
print(b.global_path.global_path.grad.data)
"""
class JoinLayerGen:
    def __init__(self, width, global_p=0.5, deepest=False):
        self.global_p = global_p
        self.width = width
        self.switch_seed = np.random.randint(1, int(1e6))
        self.path_seed = np.random.randint(1, int(1e6))
        self.deepest = deepest
        if deepest:
            self.is_global = True
            self.path_array =torch.tensor([1.] + [.0]*(width - 1),dtype=torch.float64)
        else:
            self.is_global = self._build_global_switch()
            self.path_array = self._build_global_path_arr()

    def _build_global_path_arr(self):
        # The path the block will take when using global droppath
        return rand_one_in_array(seed=self.path_seed, count=self.width)

    def _build_global_switch(self):
        # A randomly sampled tensor that will signal if the batch
        # should use global or local droppath
        torch.manual_seed(self.switch_seed)
        return torch.distributions.Binomial(1, torch.tensor([self.global_p], dtype=torch.float64).expand(1)).sample().item()==True
    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        return JoinLayer(drop_p=drop_p, is_global=global_switch, global_path=global_path, force_path=self.deepest)
"""
A=JoinLayerGen(5,deepest=True)
B=A.get_join_layer(0.3)
a=torch.autograd.Variable(torch.tensor([[[5,3,1],[4,3,2],[3,3,3]],[[5,3,1],[4,6,2],[3,4,3]]],dtype=torch.float64),requires_grad=True)
print(B(a))r
"""
class fractal_conv(nn.Module):
    def __init__(self,in_filter,filter,nb_col,dropout=None):
        super(fractal_conv, self).__init__()
        self.check=dropout
        self.conv= nn.Conv2d(in_filter,filter, kernel_size=nb_col, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filter, filter, kernel_size=nb_col, stride=1, padding=1)
        self.dropout=nn.Dropout2d((dropout if dropout else float(0.2)))
        self.bn=nn.BatchNorm2d(filter)
        self.filter=filter
        self.relu=nn.ReLU()

    def forward(self,x):
        if x.shape[1]==self.filter:
            x=self.conv2(x)
        else:
            x=self.conv(x)
        if self.check:
            x=self.dropout(x)
        x=self.bn(x)
        x=self.relu(x)
        return x
class fractal_block(nn.Module):
    def __init__(self,join_gen,c,in_filter,filter,nb_col,nb_row,drop_p,dropout=None):
        super(fractal_block, self).__init__()
        self.columns=lambda z:[[z.cuda()] for _ in range(c)]
        self.fractal_conv=fractal_conv(in_filter,filter,nb_col,dropout)
        self.merged= join_gen.get_join_layer(drop_p=drop_p)
        self.c=c
        self.nb_col=nb_col
        self.in_filter=in_filter
        self.filter=filter
        self.nb_row=nb_row
        self.drop_p=drop_p
        self.dropout=dropout

    def forward(self,x):
        x=self.columns(x)
        for row in range(2**(self.c-1)):
            t_row=[]
            for col in range(self.c):
                prop=2**(col)
                if (row+1)%prop==0:
                    x[col][-1]=self.fractal_conv(x[col][-1])
                    t_row.append(col)
            if len(t_row)>1:
                merging=torch.tensor([x[i][-1].cpu().detach().numpy() for i in t_row],dtype=torch.float32).cuda()
                merged = self.merged(merging)
                for i in t_row:
                    x[i][-1]=merged
        return x[0][-1]
class fractal_net(nn.Module):
    def __init__(self,b,c,conv,drop_path,global_p=0.5,dropout=None,deepest=False):
        super(fractal_net,self).__init__()
        self.b=b
        self.c=c
        self.conv=conv
        self.drop_path=drop_path
        self.global_p=global_p
        self.dropout=dropout
        self.deepest=deepest
        self.joinlayergen= JoinLayerGen(width=c, global_p=global_p, deepest=deepest)
        self.fractal_net=nn.Sequential(*[j for i,(in_filter,filter, nb_col, nb_row) in myenumerate(conv,b) for j in [fractal_block(
                                 join_gen=self.joinlayergen,
                                 c=c,
                                 in_filter=in_filter,
                                 filter=filter,
                                 nb_col=nb_col,
                                 nb_row=nb_row,
                                 drop_p=drop_path,
                                 dropout=dropout[i] if dropout else None),nn.MaxPool2d((2,2),stride=(2,2))]])

    def forward(self,x):
        global output
        output=x
        output=self.fractal_net(output)
        return output

