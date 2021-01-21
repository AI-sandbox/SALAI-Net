import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class LAINetOriginal(nn.Module):

    def __init__(self, input_dimension, ancestry_number, window_size=500, hidden_size=30,
                 list_dropout_p=[0.0, 0.0], with_bias=True, is_haploid=False, 
                 norm_layer=nn.BatchNorm1d, activation=nn.ReLU(), smooth_kernel_size = 75,  
                 with_softmax_before_smooth=True, normalize_input_smoother=False):
        
        super(LAINetOriginal, self).__init__()
            
        self.input_dimension = input_dimension
        self.num_ancestries = ancestry_number
        self.dim_list = [window_size, hidden_size, ancestry_number] 
        self.windows_size = window_size
        self.num_windows = int(np.floor(self.input_dimension/self.windows_size))

        self.is_haploid = is_haploid

        self.with_bias = with_bias
        self.norm_layer = norm_layer
        self.activation = activation
        

        self.smooth_kernel_size = smooth_kernel_size
        self.with_softmax_before_smooth = with_softmax_before_smooth
        

      
        ## Network architecture
        ## Base Windows            
        self.ListMLP = nn.ModuleList()

        for i in range(self.num_windows):
            if i == self.num_windows-1:
                dim_list = np.copy(self.dim_list)
                dim_list[0] += np.remainder(self.input_dimension, self.windows_size)
            else:
                dim_list = np.copy(self.dim_list)

            self.ListMLP.append(self.gen_mlp(dim_list))
           
    
        ## Smoother
        self.smoother = self.get_smoother()


        
    def input2windows(self, x):
        windows = []
        for j in range(len(self.ListMLP)):
            if j == len(self.ListMLP)-1:
                _x = x[:, j*self.windows_size:]
            else:
                _x = x[:, j * self.windows_size : (j + 1) * self.windows_size]
            windows.append(_x)
        return windows
    
    
    def labels2windows(self, x):
        wins = self.input2windows(x)
        vs = []
        for w in wins:
            v, i = torch.mode(w, axis=1)
            vs.append(v)
        vmap = torch.stack(vs, axis=1)
        return vmap

    
    
    def forward(self, x):
        if self.is_haploid:
            return self.forward_haploid(x)
        else:
            return self.forward_diploid(x)

               
    
    def forward_base(self, x):
        windows = self.input2windows(x)
        outs = []
        for j in range(len(self.ListMLP)):
            o = self.ListMLP[j](windows[j])
            outs.append(o)
        out = torch.stack(outs, dim=2)

        return out
    
    
    def forward_smoother(self, x):
        out = x
        
        if self.with_softmax_before_smooth:
            _out = F.softmax(out, dim=1)
        else:
            _out = out

        out_smooth = self.smoother(_out)



        if not self.is_haploid:
            out_smooth = out_smooth[:,:,:,0:2]
                
        return out_smooth
    
    
    def forward_haploid(self, x):


        out_base = self.forward_base(x)
        out_smooth = self.forward_smoother(out_base)

        out_smooth = F.interpolate(out_smooth, size=self.input_dimension)
        out_smooth = out_smooth.permute(0, 2, 1)

        return out_base, out_smooth
    
    
    def forward_diploid(self, x):
        out_0 = self.forward_base(x[:,:,0])
        out_1 = self.forward_base(x[:,:,1])
        out_base = torch.stack([out_0, out_1], dim=3)

        out_smooth = self.forward_smoother(out_base)
        return out_base, out_smooth
    
    
    def gen_mlp(self, list_dim):
        _layer_list = []
        for i in range(len(list_dim)-1):
            _layer_list.append(nn.Linear(list_dim[i], list_dim[i+1], bias=self.with_bias))
            if i < len(list_dim)-2:
                    _layer_list.append(self.activation)
                    _layer_list.append(self.norm_layer(list_dim[i+1], affine=False))

        mlp = nn.Sequential(*_layer_list)
        return mlp



    def get_smoother(self):
        return self._get_conv(self.num_ancestries, self.num_ancestries)
        
        
    def _get_conv(self, in_dim, out_dim):
        if self.is_haploid:
            return nn.Conv1d(in_dim, out_dim, self.smooth_kernel_size, padding=int(np.floor(self.smooth_kernel_size / 2)),
                                 padding_mode='reflect')
        else:
            return nn.Conv2d(in_dim, out_dim, (self.smooth_kernel_size, 2), padding=(int(np.floor(self.smooth_kernel_size / 2)),1),
                             padding_mode='reflect')








