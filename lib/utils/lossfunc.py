import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from functools import reduce
import scipy.sparse as sp
import numpy as np
from chumpy.utils import row, col

# https://github.com/google-research/jax3d/blob/6a8913e6678c994d08e84965609adbeb31cde0ae/jax3d/projects/generative/nerf/losses.py#L178
# def hard_surface(sample_weights: FloatArray[...],
#                  weight: float = 0.0) -> Tuple[FloatArray, float]:
#   """Hard surface density regularizer loss.
#   Args:
#     sample_weights: Per-sample contribution weights from volume rendering.
#     weight: The scalar weight controlling the strength of the loss.
#   Returns:
#     loss: The scalar loss value with no weight applied.
#     weight: The scalar weight controlling the strength of the loss.
#   """
#   loss = jnp.mean(-jnp.log(
#       jnp.exp(-jnp.abs(sample_weights)) +
#       jnp.exp(-jnp.abs(1.0 - sample_weights))))
#   return loss, weight

def hard_loss_func(x, scale=1.):
    y = -torch.log(torch.exp(-scale*torch.abs(x)) + torch.exp(-scale*torch.abs(1-x)))
    # import ipdb; ipdb.set_trace()
    return y.mean()

def image_loss(pred, gt, mask=None, losstype=F.smooth_l1_loss):
    if mask is not None:
        pred = pred * mask
        gt = gt * mask
        loss = losstype(pred, gt, reduction='none').sum([1,2,3]) / mask.sum([1,2,3])
        loss = loss.mean()
    else:
        loss = losstype(pred, gt, reduction='mean')
    return loss
                
def get_vert_connectivity(num_vertices, faces):
    """
    Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12.
    Adapted from https://github.com/mattloper/opendr/
    """

    vpv = sp.csc_matrix((num_vertices,num_vertices))

    # for each column in the faces...
    for i in range(3):
        IS = faces[:,i]
        JS = faces[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T
    return vpv

def get_vertices_per_edge(num_vertices, faces):
    """
    Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()
    Adapted from https://github.com/mattloper/opendr/
    """

    vc = sp.coo_matrix(get_vert_connectivity(num_vertices, faces))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]] # for uniqueness
    return result

def relative_edge_loss(vertices1, vertices2, vertices_per_edge=None, faces=None, lossfunc=torch.nn.functional.mse_loss):
    """
    Given two meshes of the same topology, returns the relative edge differences.

    """

    if vertices_per_edge is None and faces is not None:
        vertices_per_edge = get_vertices_per_edge(len(vertices1), faces)
    elif vertices_per_edge is None and faces is None:
        raise ValueError("Either vertices_per_edge or faces must be specified")

    edges_for = lambda x: x[:, vertices_per_edge[:, 0], :] - x[:, vertices_per_edge[:, 1], :]
    return lossfunc(edges_for(vertices1), edges_for(vertices2))


def relative_laplacian_loss(mesh1, mesh2, lossfunc=torch.nn.functional.mse_loss):
    L = mesh2.laplacian_packed()
    L1 = L.mm(mesh1.verts_packed())
    L2 = L.mm(mesh2.verts_packed())
    loss = lossfunc(L1, L2)
    return loss    
    
def mesh_laplacian(meshes, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()
    
    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    # import ipdb; ipdb.set_trace()
    # loss = loss.norm(dim=1)

    # loss = loss * weights
    # return loss.sum() / N
    return loss

    

def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    # import ipdb; ipdb.set_trace()
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    # if mask is not None:
    #     loss = loss.abs().sum()/mask.sum()
    # else:
    loss = loss.abs().mean()
    return loss
    
class MSELoss(nn.Module):
    def __init__(self, w_alpha=0.):
        super(MSELoss, self).__init__()
        # self.loss = nn.MSELoss(reduction='mean')
        self.w_alpha = w_alpha

    def forward(self, inputs, targets):
        # import ipdb; ipdb.set_trace()
        # print(inputs['rgb_coarse'].min(), inputs['rgb_coarse'].max(), targets.min(), targets.max())
        # loss = self.loss(inputs['rgb_coarse'], targets[:,:3])
        loss = huber(inputs['rgb_coarse'], targets[:,:3])
        
        if 'rgb_fine' in inputs:
            loss += huber(inputs['rgb_fine'], targets[:,:3])
        
        # import ipdb; ipdb.set_trace()
        if self.w_alpha>0. and targets.shape[1]>3:
            weights = inputs['weights_coarse']
            pix_alpha = weights.sum(dim=1)
            # pix_alpha[pix_alpha>1] = 1
            # loss += self.loss(pix_alpha, targets[:,3])*self.w_alpha
            loss += huber(pix_alpha, targets[:,3])*self.w_alpha
            if 'weights_fine' in inputs:
                weight = inputs['weights_fine']
                pix_alpha = weights.sum(dim=1)
                # loss += self.loss(pix_alpha, targets[:,3])*self.w_alpha
                loss += huber(pix_alpha, targets[:,3])*self.w_alpha

        return loss
               

loss_dict = {'mse': MSELoss}


### IDMRF loss
class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x/self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        # print([x for x in out])
        return out

# Ref: https://github.com/shepnerd/inpainting_gmcnn/blob/ba7f7109c38c3805800283cdb9d79cd7c4a3294f/pytorch/model/loss.py
class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist)/self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss

        # loss = 0
        # for key in self.feat_style_layers.keys():
        #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
        # return loss


##############################################
## ref: https://github.com/cydonia999/VGGFace2-pytorch
from .frnet import resnet50, load_state_dict
class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_model, pretrained_data='vggface2'):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval().cuda()
        load_state_dict(self.reg_model, pretrained_model)
        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).cuda()

    def reg_features(self, x):
        # out = []
        margin=10
        x = x[:,:,margin:224-margin,margin:224-margin]
        # x = F.interpolate(x*2. - 1., [224,224], mode='nearest')
        x = F.interpolate(x*2. - 1., [224,224], mode='bilinear')
        # import ipdb; ipdb.set_trace()
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # import ipdb;ipdb.set_trace()
        img = img[:, [2,1,0], :, :].permute(0,2,3,1) * 255 - self.mean_bgr
        img = img.permute(0,3,1,2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        # loss = ((gen_out - tar_out)**2).mean()
        loss = self._cos_metric(gen_out, tar_out).mean()
        return loss


''' Loss functions from DECA https://github.com/YadiraF/DECA/blob/2d8864b34120d5a2c608dbce9aa5eec76e90cef9/decalib/utils/lossfunc.py#L195
'''
def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        real_2d_kp[:,:,2] = weights[None,:]*real_2d_kp[:,:,2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k

def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight

def eye_dis(landmarks):
    # left eye:  [38,42], [39,41] - 1
    # right eye: [44,48], [45,47] -1
    eye_up = landmarks[:,[37, 38, 43, 44], :]
    eye_bottom = landmarks[:,[41, 40, 47, 46], :]
    dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]
    return dis

def eyed_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    pred_eyed = eye_dis(predicted_landmarks[:,:,:2])
    gt_eyed = eye_dis(real_2d[:,:,:2])

    loss = (pred_eyed - gt_eyed).abs().mean()
    return loss

def lip_dis(landmarks):
    # up inner lip:  [62, 63, 64] - 1
    # down innder lip: [68, 67, 66] -1
    lip_up = landmarks[:,[61, 62, 63], :]
    lip_down = landmarks[:,[67, 66, 65], :]
    dis = torch.sqrt(((lip_up - lip_down)**2).sum(2)) #[bz, 4]
    return dis

def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    pred_lipd = lip_dis(predicted_landmarks[:,:,:2])
    gt_lipd = lip_dis(real_2d[:,:,:2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss
    
def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    #smaller inner landmark weights
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # import ipdb; ipdb.set_trace()
    real_2d = landmarks_gt
    weights = torch.ones((68,)).cuda()
    weights[5:7] = 2
    weights[10:12] = 2
    # nose points
    weights[27:36] = 1.5
    weights[30] = 3
    weights[31] = 3
    weights[35] = 3
    # inner mouth
    weights[60:68] = 1.5
    weights[48:60] = 1.5
    weights[48] = 3
    weights[54] = 3

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight

def landmark_loss_tensor(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight


import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


