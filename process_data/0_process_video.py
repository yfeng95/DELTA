from ipaddress import ip_address
import os, sys
import argparse
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
import shutil
import torch
import cv2
import numpy as np
from PIL import Image

def plot_points(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (n, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()
    kpts = kpts.copy()
    kpts = kpts.astype(np.int32)

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, c, 2)  
    return image

def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im
        
def generate_frame(inputpath, savepath, subject_name=None):
    ''' extract frames from video or copy frames from image folder
    '''
    os.makedirs(savepath, exist_ok=True)
    if subject_name is None:
        subject_name = Path(inputpath).stem
    ## video data
    if os.path.isfile(inputpath) and (os.path.splitext(inputpath)[-1] in ['.mp4', '.csv', '.MOV']):
        logger.info(f'extract frames from video: {inputpath}...')        
        vidcap = cv2.VideoCapture(inputpath)
        count = 0
        success, image = vidcap.read()
        cv2.imwrite(os.path.join(savepath, f'{subject_name}_f{count:06d}.png'), image) 
        while success:
            count += 1
            success,image = vidcap.read()
            if image is None:
                break
            imagepath = os.path.join(savepath, f'{subject_name}_f{count:06d}.png')
            cv2.imwrite(imagepath, image)     # save frame as JPEG png
    elif os.path.isdir(inputpath):
        logger.info(f'copy frames from folder: {inputpath}...')
        imagepath_list = glob(inputpath + '/*.jpg') +  glob(inputpath + '/*.png') + glob(inputpath + '/*.jpeg')
        imagepath_list = sorted(imagepath_list)
        for count, imagepath in enumerate(imagepath_list):
            shutil.copyfile(imagepath, os.path.join(savepath, f'{subject_name}_f{count:06d}.png'))
        print('frames are stored in {}'.format(savepath))
    else:
        logger.info(f'please check the input path: {inputpath}')
        exit()
    logger.info(f'video frames are stored in {savepath}')


def generate_image(inputpath, savepath, subject_name=None, crop=False, crop_each=False, image_size=512, scale_bbox=100., device='cuda:0'):
    ''' generate image from given frame path. 
    run face detection and face parsing to filter bad frames
    '''
    logger.info(f'generae images, crop {crop}, image size {image_size}')
    os.makedirs(savepath, exist_ok=True)
    if subject_name is None:
        subject_name = inputpath.split('/')[-2]
    import face_alignment
    # detect_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
    from fdlite import (
        FaceDetection,
        FaceLandmark,
        face_detection_to_roi,
        IrisLandmark,
        iris_roi_from_face_landmarks,
        )
    detect_faces = FaceDetection()
    detect_face_landmarks = FaceLandmark()
    if os.path.isdir(inputpath):
        imagepath_list = glob(inputpath + '/*.jpg') +  glob(inputpath + '/*.png') + glob(inputpath + '/*.jpeg')
        imagepath_list = sorted(imagepath_list)
        # if crop, detect the bbox of the first image and use the bbox for all frames
        if crop:
            imagepath = imagepath_list[0]
            logger.info(f'detect first image {imagepath}')
            import face_alignment
            detect_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
            image = imread(imagepath)
            out = detect_model.get_landmarks(image)
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            np.savetxt(os.path.join(Path(inputpath).parent, 'image_bbox.txt'), bbox)
            ## calculate warping function for image cropping
            old_size = max(right - left, bottom - top)
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*scale_bbox)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            DST_PTS = np.array([[0,0], [0,image_size - 1], [image_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        valid_count = 0
        for count, imagepath in enumerate(tqdm(imagepath_list)):
            if crop:
                image = imread(imagepath)
                if crop_each:
                    import face_alignment
                    detect_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
                    out = detect_model.get_landmarks(image)
                    kpt = out[0].squeeze()
                    left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                    top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                    bbox = [left,top, right, bottom]
                    np.savetxt(os.path.join(Path(inputpath).parent, 'image_bbox.txt'), bbox)
                    ## calculate warping function for image cropping
                    old_size = max(right - left, bottom - top)
                    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
                    size = int(old_size*scale_bbox)
                    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
                    DST_PTS = np.array([[0,0], [0,image_size - 1], [image_size - 1, 0]])
                    tform = estimate_transform('similarity', src_pts, DST_PTS)
                dst_image = warp(image, tform.inverse, output_shape=(image_size, image_size))
                imsave(os.path.join(savepath, f'{subject_name}_f{count:06d}.png'), dst_image)
            else:
                
                ##-- filter images with no face or multiple faces
                # img = Image.open(imagepath)
                # face_detections = detect_faces(img)
                # img_size = img.size
                # if len(face_detections) != 1:
                #     print("Empty face")
                #     continue
                # try:
                #     face_roi = face_detection_to_roi(face_detections[0], img_size)
                # except ValueError:
                #     continue
                # face_landmarks = detect_face_landmarks(img, face_roi)
                # if len(face_landmarks) == 0:
                #     print("Empty iris landmarks")
                #     continue
                image = imread(imagepath)
                h, w, _ = image.shape
                if h!=image_size or w!=image_size:
                    dst_image = resize(image, [image_size, image_size])
                    dst_image = (dst_image*255).astype(np.uint8)
                    imsave(os.path.join(savepath, f'{subject_name}_f{count:06d}.png'), dst_image)
                else:
                    shutil.copyfile(imagepath, os.path.join(savepath, f'{subject_name}_f{count:06d}.png'))
    logger.info(f'images are stored in {savepath}')

def generate_matting_rvm(inputpath, savepath, ckpt_path='assets/rvm/rvm_resnet50.pth', device='cuda:0'):
    sys.path.append('./submodules/RobustVideoMatting')
    from model import MattingNetwork
    EXTS = ['jpg', 'jpeg', 'png']
    segmentor = MattingNetwork(variant='resnet50').eval().to(device)
    segmentor.load_state_dict(torch.load(ckpt_path))

    images_folder = inputpath
    output_folder = savepath
    os.makedirs(output_folder, exist_ok=True)

    frame_IDs = os.listdir(images_folder)
    frame_IDs = [id.split('.')[0] for id in frame_IDs if id.split('.')[-1] in EXTS]
    frame_IDs.sort()
    frame_IDs = frame_IDs[:4][::-1] + frame_IDs

    rec = [None] * 4                                       # Initial recurrent 
    downsample_ratio = 1.0                                 # Adjust based on your video.   

    # bgr = torch.tensor([1, 1, 1.]).view(3, 1, 1).cuda() 
    for i in tqdm(range(len(frame_IDs))):
        frame_ID = frame_IDs[i]
        img_path = os.path.join(images_folder, '{}.png'.format(frame_ID))
        try:
            img_masked_path = os.path.join(output_folder, '{}.png'.format(frame_ID))
            img = cv2.imread(img_path)
            src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(src).float() / 255.
            src = src.permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                fgr, pha, *rec = segmentor(src.to(device), *rec, downsample_ratio)  # Cycle the recurrent states.
            pha = pha.permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
            # check the difference of current 
            mask = (pha * 255).astype(np.uint8)
            img_masked = np.concatenate([img, mask], axis=-1)
            cv2.imwrite(img_masked_path, img_masked)
        except:
            os.remove(img_path)
            logger.info(f'matting failed for image {img_path}, delete it')
            
    sys.modules.pop('model')
    
# better for portrait
def generate_matting_MODNet(inputpath, savepath, ckpt_path='assets/MODNet/modnet_webcam_portrait_matting.ckpt', device='cuda:0'):
    sys.path.append('./submodules/MODNet')
    from src.models.modnet import MODNet
    import torchvision.transforms as transforms
    import torch.nn as nn
    from PIL import Image

    EXTS = ['jpg', 'jpeg', 'png']
    modnet = MODNet(backbone_pretrained=False).to(device)
    modnet = nn.DataParallel(modnet)
    modnet.load_state_dict(torch.load(ckpt_path))
    modnet.eval()
    torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )
    rh = 512
    rw = 512
    rh = rh - rh % 32
    rw = rw - rw % 32
    
    images_folder = inputpath
    output_folder = savepath
    os.makedirs(output_folder, exist_ok=True)

    frame_IDs = os.listdir(images_folder)
    frame_IDs = [id.split('.')[0] for id in frame_IDs if id.split('.')[-1] in EXTS]
    frame_IDs.sort()
    frame_IDs = frame_IDs[:4][::-1] + frame_IDs

    rec = [None] * 4                                       # Initial recurrent 
    downsample_ratio = 1.0                                 # Adjust based on your video.   

    # bgr = torch.tensor([1, 1, 1.]).view(3, 1, 1).cuda() 
    for i in tqdm(range(len(frame_IDs))):
        frame_ID = frame_IDs[i]
        img_path = os.path.join(images_folder, '{}.png'.format(frame_ID))
        img_masked_path = os.path.join(output_folder, '{}.png'.format(frame_ID))
        img = cv2.imread(img_path)[:,:,:3]
        h, w, _ = img.shape
        frame_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_np = cv2.resize(frame_np, (rw, rh), cv2.INTER_AREA)
        # frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
        frame_PIL = Image.fromarray(frame_np)
        frame_tensor = torch_transforms(frame_PIL)
        frame_tensor = frame_tensor[None, :, :, :]
        frame_tensor = frame_tensor.to(device)
        with torch.no_grad():
            _, _, matte_tensor = modnet(frame_tensor, True)
        matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
        matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
        mask = (matte_np[:,:,[0]] * 255).astype(np.uint8)
        img_masked = np.concatenate([img[...,:3], mask], axis=-1)
        cv2.imwrite(img_masked_path, img_masked)

def generate_landmark2d(inputpath, savepath, device='cuda:0', vis=False):
    logger.info(f'generae 2d landmarks')
    os.makedirs(savepath, exist_ok=True)
    import face_alignment
    detect_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
    
    imagepath_list = glob(os.path.join(inputpath, '*.png'))
    imagepath_list = sorted(imagepath_list)
    for imagepath in tqdm(imagepath_list):
        name = Path(imagepath).stem
        image = imread(imagepath)[:,:,:3]
        try: 
            out = detect_model.get_landmarks(image)
            kpt = out[0].squeeze()
            np.savetxt(os.path.join(savepath, f'{name}.txt'), kpt)
            if vis:
                image = cv2.imread(imagepath)
                image_point = plot_points(image, kpt)
                cv2.imwrite(os.path.join(savepath, f'{name}.png'), image_point)
        except:
            os.remove(imagepath)
            logger.info(f'2D landmark detection filed with image {imagepath}, delete it')
              
def generate_landmark3d(inputpath, savepath, device='cuda:0', vis=False):
    logger.info(f'generae 3d landmarks')
    os.makedirs(savepath, exist_ok=True)
    import face_alignment
    detect_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device, flip_input=False)
    
    imagepath_list = glob(os.path.join(inputpath, '*.png'))
    imagepath_list = sorted(imagepath_list)
    for imagepath in tqdm(imagepath_list):
        name = Path(imagepath).stem
        image = imread(imagepath)[:,:,:3]
        try: 
            out = detect_model.get_landmarks(image)
            kpt = out[0].squeeze()
            np.savetxt(os.path.join(savepath, f'{name}.txt'), kpt)
            if vis:
                image = cv2.imread(imagepath)
                image_point = plot_points(image, kpt)
                cv2.imwrite(os.path.join(savepath, f'{name}.png'), image_point)
        except:
            os.remove(imagepath)
            logger.info(f'3D landmark detection filed with image {imagepath}, delete it')
            
def generate_iris(inputpath, savepath, device='cuda:0', vis=False):
    logger.info(f'generae iris detection')
    os.makedirs(savepath, exist_ok=True)
    from fdlite import (
        FaceDetection,
        FaceLandmark,
        face_detection_to_roi,
        IrisLandmark,
        iris_roi_from_face_landmarks,
        )
    # iris detector
    detect_faces = FaceDetection()
    detect_face_landmarks = FaceLandmark()
    detect_iris_landmarks = IrisLandmark()
    
    imagepath_list = glob(os.path.join(inputpath, '*.png'))
    imagepath_list = sorted(imagepath_list)
    # iris_dict = {}
    for imagepath in tqdm(imagepath_list):
        lmks_array = np.zeros([2,3], dtype=np.float32)
        name = Path(imagepath).stem
        img = Image.open(imagepath)

        width, height = img.size
        img_size = (width, height)

        face_detections = detect_faces(img)
        if len(face_detections) != 1:
            print("Empty iris landmarks")
        else:
            for face_detection in face_detections:
                try:
                    face_roi = face_detection_to_roi(face_detection, img_size)
                except ValueError:
                    print("Empty iris landmarks")
                    break

                face_landmarks = detect_face_landmarks(img, face_roi)
                if len(face_landmarks) == 0:
                    print("Empty iris landmarks")
                    break

                iris_rois = iris_roi_from_face_landmarks(face_landmarks, img_size)

                if len(iris_rois) != 2:
                    print("Empty iris landmarks")
                    break

                lmks = []
                for k, iris_roi in enumerate(iris_rois[::-1]):
                    try:
                        iris_landmarks = detect_iris_landmarks(img, iris_roi).iris[
                                         0:1
                                         ]
                    except np.linalg.LinAlgError:
                        print("Failed to get iris landmarks")
                        break

                    for landmark in iris_landmarks:
                        lmks.append([landmark.x * width, landmark.y * height])
                        
        lmks = np.array(lmks)
        if lmks.shape[0] == 2:
            lmks_array[:,:2] = lmks
            lmks_array[:, 2] = 1.
        np.savetxt(os.path.join(savepath, f'{name}.txt'), lmks_array)
        if vis:
            image = cv2.imread(imagepath)
            image_point = plot_points(image, lmks_array)
            cv2.imwrite(os.path.join(savepath, f'{name}.png'), image_point)
            
def generate_face_parsing(inputpath, savepath, ckpt_path='assets/face_parsing/model.pth', device='cuda:0', vis=False):
    logger.info(f'generae face parsing')
    os.makedirs(savepath, exist_ok=True)
    sys.path.insert(0, './submodules/face-parsing.PyTorch')
    from model import BiSeNet
    import torchvision.transforms as transforms

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    net.load_state_dict(torch.load(ckpt_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    with torch.no_grad():
        for image_path in tqdm(os.listdir(inputpath)):
            name = image_path.split('.')[0]
            img = Image.open(os.path.join(inputpath, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            image = np.array(image)
            if image.shape[-1] == 4:
                image = image/255.
                alpha_image = image[:, :, 3:]
                image = image[:,:,:3]*alpha_image + (1-alpha_image)
                image = Image.fromarray((image*255).astype(np.uint8))
                
                # image = image[:,:,:3]
            elif image.shape[-1] == 1:
                import ipdb; ipdb.set_trace()
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))
            # if len(np.unique(parsing)) < 5:
            #     os.remove(os.path.join(inputpath, image_path))
            #     logger.info(f'face parsing for image {os.path.join(inputpath, image_path)} failed, less than 5 cat was found, delete image')
            #     continue
            cv2.imwrite(os.path.join(savepath, f'{name}.png'), parsing)
            if vis:
                parsing_vis = vis_parsing_maps(image, parsing, stride=1)
                cv2.imwrite(os.path.join(savepath, f'{name}_vis.png'), parsing_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    sys.path.remove('./submodules/face-parsing.PyTorch')
    sys.modules.pop('model')
    
def generate_face_normals(inputpath, savepath, ckpt_path='assets/face_normals/model.pth', device='cuda:0'):
    logger.info(f'generae face normals')
    os.makedirs(savepath, exist_ok=True)
    sys.path.append('./submodules')
    from face_normals.resnet_unet import ResNetUNet
    from torchvision.transforms import Compose, ToTensor
    img_transform = Compose([
            ToTensor()
            ])
    
    model = ResNetUNet(n_class = 3).cuda()
    model.load_state_dict(torch.load(ckpt_path))                              
    model.eval()
    from torch.autograd import Variable
    import face_alignment
    detect_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
    
    imagepath_list = glob(os.path.join(inputpath, '*.png'))
    imagepath_list = sorted(imagepath_list)
    # iris_dict = {}
    for imagepath in tqdm(imagepath_list):
        name = Path(imagepath).stem
        # load image
        img = imread(imagepath)[:,:,:3]
       
        # detect kp
        out = detect_model.get_landmarks(img)
        kpt = out[0].squeeze()
            
        # crop & resize
        umin = np.min(kpt[:,0]) 
        umax = np.max(kpt[:,0])
        vmin = np.min(kpt[:,1]) 
        vmax = np.max(kpt[:,1])
        umean = np.mean((umin,umax))
        vmean = np.mean((vmin,vmax))
        l = round( 1.2 * np.max((umax-umin,vmax-vmin)))
        if (l > np.min(img.shape[:2])):
            l = np.min(img.shape[:2])
        us = round(np.max((0,umean-float(l)/2)))
        ue = us + l
        vs = round(np.max((0,vmean-float(l)/2))) 
        ve = vs + l 
        if (ue>img.shape[1]):
            ue = img.shape[1]
            us = img.shape[1]-l
        if (ve>img.shape[0]):
            ve = img.shape[0]
            vs = img.shape[0]-l
                        
        us = int(us)
        ue = int(ue)  
            
        vs = int(vs)
        ve = int(ve)    
            
        crop_img = resize(img[vs:ve,us:ue],(256,256))
        crop_img = (crop_img*255).astype(np.uint8)
        # msk = cv2.resize(msk[vs:ve,us:ue],(256,256),interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('tmp.jpg', crop_img)
        # crop_img = Image.open('tmp.jpg')
        # import ipdb; ipdb.set_trace()
        img_tensor = img_transform(crop_img).unsqueeze(0)  
        img_tensor = Variable(img_tensor.cuda())
        
        outs = model(img_tensor)[0]  
        out = np.array(outs[0].data.permute(1,2,0).cpu())  
        out = out / np.expand_dims(np.sqrt(np.sum(out * out, 2)),2)
        out = 127.5 * (out + 1.0)
        out_orig = np.zeros_like(img, dtype=np.uint8)*255
        out_orig[vs:ve, us:ue] = resize(out, [ve-vs, ue-us])
        imsave(os.path.join(savepath, f'{name}.png'), out_orig)            

class DataProcessor():
    def __init__(self, args):
        # self.actions = args.actions
        self.savepath = args.savepath
        # if 'copy' in actions:
                
    def check_run(self, subject_name, action):
        outputpath = os.path.join(self.savepath, subject_name, action)
        
    def run(self, subject_path, vis=False, crop=False, crop_each=False, ignore_existing=False):
        subject_name = Path(subject_path).stem
        savepath = os.path.join(self.savepath, subject_name)
        os.makedirs(savepath, exist_ok=True)
        # 0. copy frames from video or image folder
        if ignore_existing or not os.path.exists(os.path.join(savepath, 'frame')):
            generate_frame(subject_path, os.path.join(savepath, 'frame'))
        
        # 1. crop image from frames
        if ignore_existing or not os.path.exists(os.path.join(savepath, 'image')):
            generate_image(os.path.join(savepath, 'frame'), os.path.join(savepath, 'image'), subject_name=subject_name,
                           crop=crop, crop_each=crop_each, image_size=512, scale_bbox=2.5, device='cuda:0')
        
        # 2. video matting
        if ignore_existing or not os.path.exists(os.path.join(savepath, 'matting')):
            generate_matting_MODNet(os.path.join(savepath, 'image'), os.path.join(savepath, 'matting'))
            # generate_matting_rvm(os.path.join(savepath, 'image'), os.path.join(savepath, 'matting'))
            
        # 3. landmarks
        if ignore_existing or not os.path.exists(os.path.join(savepath, 'landmark2d')):
            generate_landmark2d(os.path.join(savepath, 'image'), os.path.join(savepath, 'landmark2d'), vis=vis)
        # if ignore_existing and not os.path.exists(os.path.join(savepath, 'landmark3d')):
        #     generate_landmark3d(os.path.join(savepath, 'image'), os.path.join(savepath, 'landmark3d'), vis=vis)
        if ignore_existing or not os.path.exists(os.path.join(savepath, 'iris')):
            generate_iris(os.path.join(savepath, 'image'), os.path.join(savepath, 'iris'), vis=vis)
        
        # 4. face parsing
        if ignore_existing or not os.path.exists(os.path.join(savepath, 'face_parsing')):
            generate_face_parsing(os.path.join(savepath, 'matting'), os.path.join(savepath, 'face_parsing'), vis=vis)
        
        # 5. face normal
        # if ignore_existing and not os.path.exists(os.path.join(savepath, 'face_normals')):
        #     generate_face_normals(os.path.join(savepath, 'image'), os.path.join(savepath, 'face_normals'))
        
def main(args):    
    logger.add(args.logpath)
    if args.videopath is not None:
        subject_list = [args.videopath]
    else:
        with open(args.list, 'r') as f:
            lines = f.readlines()
        subject_list = [s.strip() for s in lines]
        if args.subject_idx is not None:
            if args.subject_idx > len(subject_list):
                print('idx error!')
            else:
                subject_list = [subject_list[args.subject_idx]]
    ## model 
    processor = DataProcessor(args)
    for subjectpath in tqdm(subject_list):
        processor.run(subjectpath, vis=args.vis, crop=args.crop, crop_each=args.crop_each, ignore_existing=args.ignore_existing)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate dataset from video or image folder')
    parser.add_argument('--list', default='lists/subject_list.txt', type=str,
                        help='path to the subject data, can be image folder or video')
    parser.add_argument('--videopath', default=None, type=str,
                        help='path to the video, if not None, then ignore the list file')
    parser.add_argument('--logpath', default='logs/generate_data.log', type=str,
                        help='path to the subject data, can be image folder or video')
    parser.add_argument('--savepath', default='/is/cluster/yfeng/Data/Projects-data/DELTA/datasets/face', type=str,
                        help='path to save processed data')
    parser.add_argument('--subject_idx', default=None, type=int,
                        help='specify subject idx, if None (default), then use all the subject data in the list') 
    parser.add_argument("--image_size", default=512, type=int,
                        help = 'image size')
    parser.add_argument("--crop", default=False, action="store_true",
                        help='whether to crop image according to the face detection')
    parser.add_argument("--crop_each", default=False, action="store_true",
                        help='whether to crop image according to the face detection')
    parser.add_argument("--vis", default=True, action="store_true",
                        help='whether to visualize labels (lmk, iris, face parsing)')
    parser.add_argument("--ignore_existing", default=False, action="store_true",
                        help='ignore existing data')
    parser.add_argument("--filter_data", default=False, action="store_true",
                        help='check labels, if it is not good, then delete image')
    # parser.add_argument("--datas", default='frame, image, landmark, iris, matting, face_parsing, face_normals', type=lambda s: [int(item) for item in s.split(',')],
    #                     help='data type need to be processed')
    args = parser.parse_args()

    main(args)

