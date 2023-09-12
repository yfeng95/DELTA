import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import Embedding

class DeRF(nn.Module):
    def __init__(self,
                 D=6, W=128,
                 freqs_xyz=10,
                 deformation_dim=0,
                 out_channels=3,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels: number of input channels for xyz (3+3*10*2=63 by default)
        skips: add skip connection in the Dth layer
        """
        super(DeRF, self).__init__()
        self.D = D
        self.W = W
        self.freqs_xyz = freqs_xyz
        self.deformation_dim = deformation_dim
        self.skips = skips

        self.in_channels = 3 + 3*freqs_xyz*2 + deformation_dim
        self.out_channels = out_channels

        self.encoding_xyz = Embedding(3, freqs_xyz)

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+self.in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)

        self.out = nn.Linear(W, self.out_channels)

    def forward(self, xyz, deformation_code=None):
        xyz_encoded = self.encoding_xyz(xyz)
        
        if self.deformation_dim > 0:
            xyz_encoded = torch.cat([xyz_encoded, deformation_code], -1)
        
        xyz_ = xyz_encoded
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([xyz_encoded, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        out = self.out(xyz_)

        return out

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 freqs_xyz=10, freqs_dir=4,
                 use_view=True, use_normal=False,
                 deformation_dim=0, appearance_dim=0,
                 skips=[4], actvn_type='relu'):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.freqs_xyz = freqs_xyz
        self.freqs_dir = freqs_dir
        self.deformation_dim = deformation_dim
        self.appearance_dim = appearance_dim
        self.skips = skips
        self.use_view = use_view
        self.use_normal = use_normal

        self.encoding_xyz = Embedding(3, freqs_xyz)
        if self.use_view:
            self.encoding_dir = Embedding(3, freqs_dir)

        self.in_channels_xyz = 3 + 3*freqs_xyz*2 + deformation_dim

        self.in_channels_dir = appearance_dim
        if self.use_view:
            self.in_channels_dir += 3 + 3*freqs_dir*2
        if self.use_normal:
            self.in_channels_dir += 3

        if actvn_type == 'relu':
            actvn = nn.ReLU(inplace=True)
        elif actvn_type == 'leaky_relu':
            actvn = nn.LeakyReLU(0.2, inplace=True)
        elif actvn_type == 'softplus':
            actvn = nn.Softplus(beta=100)
        else:
            assert NotImplementedError

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, actvn)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+self.in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, xyz, cond=None):
        """
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
        Outputs:
            out: (B, 4), rgb and sigma
        """
        shape = xyz.shape
        # xyz = xyz.view(-1, shape[-1])
        sigma, xyz_encoding_final = self.query_density(xyz, only_sigma=False)

        dir_encoding_input = xyz_encoding_final
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        return rgb, sigma

    def query_density(self, xyz, cond=None, only_sigma=True):

        xyz_encoded = self.encoding_xyz(xyz)
        xyz_ = xyz_encoded
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([xyz_encoded, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)

        if only_sigma:
            return sigma
        
        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        return sigma, xyz_encoding_final

    def get_normal(self, xyz, deformation_code=None, delta=0.02):
        with torch.set_grad_enabled(True):
            xyz.requires_grad_(True)
            sigma = self.get_sigma(xyz, deformation_code=deformation_code, only_sigma=True)
            alpha = 1 - torch.exp(-delta * torch.relu(sigma))
            normal = torch.autograd.grad(
                outputs=alpha,
                inputs=xyz,
                grad_outputs=torch.ones_like(alpha, requires_grad=False, device=alpha.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

        return normal



