import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim


# ICGN
# the following was inspired by the initial experiment conducted by the paper introducing the ICGN
# https://colab.research.google.com/drive/1BvFXpouwmZtusB_bJKbBRuCqlcozbWva?usp=sharing
class I_CGN(nn.Module):
    """
    A PyTorch implementation of the I_CGN (Input Convex Gradient Network) class.
    
    Attributes:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Number of units in the hidden layers.
        output_dim (int): Dimensionality of the output, matching input_dim by default.
        lin (torch.nn.ModuleList): A list of linear layers for input and hidden transformations.
        act (torch.nn.Sigmoid): Sigmoid activation function used across layers.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, **kwargs):

        super(I_CGN,self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = input_dim

        
        self.lin = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim,bias=True),
                *[nn.Linear(hidden_dim,hidden_dim) for _ in range(num_layers - 1)],
        ])

        self.act = nn.Sigmoid()#lambda x: nn.Softplus()(x) - np.log(2)

    def jvp(self, x,v):
        with torch.enable_grad():
            #computes w = V(x)v 
            w = torch.autograd.functional.jvp(self.forward_, inputs=x, v=v, create_graph=True)[1]
            #compute w = V.T(x)w
            w = torch.autograd.functional.vjp(self.forward_,inputs=x,v=w, create_graph=True)[1]
        return w

    
    def forward_(self,x):
        a = self.act
        y = a(self.lin[0](x))

        for lay in self.lin[1:-1]:
            y = a(lay(y))
        
        return self.lin[-1](y) if len(self.lin) > 1 else y
        
        # for lay in self.lin[1:][::-1]:
        #     y = a(F.linear(y,lay.weight.data.T))

        
        # return F.linear(y,self.lin[0].weight.data.T)
    
    def checkjac(self,simp=False,N=100):
        x = torch.rand(1,self.input_dim)
        print("computing jacobian using autograd on forward")
        if self.input_dim < 10:
            if simp:
                M = torch.autograd.functional.jacobian(self.forward_simp,inputs=x,vectorize=True).squeeze()
            else:
                M = torch.autograd.functional.jacobian(lambda x: self.forward(x,N=N),inputs=x,vectorize=True).squeeze()
            print(M)

        print("computing jacobian using autograd on forward_")
        if self.input_dim < 10:
            V = torch.autograd.functional.jacobian(self.forward_,inputs=x,vectorize=True).squeeze()
            V = V.T@V 
            print(V)

        print("checking PSD")
        M.detach()
        with torch.no_grad():
            for _ in range(100):
                v = torch.rand(self.input_dim)
                if (v*M@v).sum() < -1e-2:
                    print("Failed PSD test", (v*M@v).sum(), "vector", v, "at point", x)
                    return
        print("passed psd test")
        return 

    #numerically computes the line integral 
    def forward(self,x,N=100,**kwargs):
        pts = torch.rand(size=(N,)).reshape(1,-1,1).to(x.device)

        in_size = x.shape[0]
        input_dim = (self.input_dim,)
        output_dim = (self.output_dim,)

        z = x.unsqueeze(1) * pts
        v = x.unsqueeze(1) * torch.ones_like(pts)

        z = z.reshape(-1,*input_dim)
        v = v.reshape(-1,*input_dim)


        #this computes the integral
        y = self.jvp(z,v)
        y = y.reshape(in_size,-1,*output_dim).sum(dim=1) / N 

        return y

    #uses simpsons rule for computing integral, more accurate but problematic because eval points are fixed
    #can't be used for learning because the model learns to avoid those points and do weird things
    #however, is very accurate at inference time
    def forward_simp(self, x, w=None,**kwargs):
        pts = torch.tensor([0,1/3,2/3,1]).reshape(1,-1,1).to(x.device)


        scale = torch.tensor([1,3,3,1]).reshape(1,-1,1).to(x.device)

        in_size = x.shape[0]
        input_dim = (self.input_dim,)
        output_dim = (self.output_dim,)

        z = x.unsqueeze(1) * pts
        if w is None: v = x.unsqueeze(1) * torch.ones_like(pts) #probably a better way to do this
        else: v = w.unsqueeze(1) * torch.ones_like(pts)
        #print(z.shape,v.shape)
        #not necessary for linear num_layers, but will be for convolutions so good practice
        z = z.reshape(-1,*input_dim)
        v = v.reshape(-1,*input_dim)

        #this computes the integral
        y = self.jvp(z,v)
        #print("after jvp", y.reshape(in_size,-1,*output_dim).shape)
        y = 1/8 * (y.reshape(in_size,-1,*output_dim)*scale).sum(dim=1)


        #print("output", y.shape)
        return y