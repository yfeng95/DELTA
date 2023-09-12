<!-- PROJECT LOGO -->

<p align="center">
  <h1 align="center">DELTA: Learning Disentangled Avatars with Hybrid 3D Representations
 </h1>
<!--  <p align="center">
    <a href="https://ps.is.tuebingen.mpg.de/person/yxiu"><strong>Yao Feng</strong></a>
    ·
    <a href="https://ps.is.tuebingen.mpg.de/person/jyang"><strong>Jinlong Yang</strong></a>
    ·
    <a href="https://ps.is.tuebingen.mpg.de/person/black"><strong>Michael J. Black</strong></a>
    .
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
    .
    <a href="https://ps.is.mpg.de/person/tbolkart"><strong>Timo Bolkart</strong></a>
  </p>
  <h2 align="center">SIGGRAPH Asia 2022 conference </h2>
  -->
  <div align="center">
    <img src="../Doc/images/data_teaser.png" alt="visualize data" width="100%">
     <!-- <h5 align="center">cropped image, subject segmentation, clothing segmentation, SMPL-X estimation </h5> -->
  </div>
</p> 
This is the script to process face (upper-body) video data for DELTA.

To process full body video, please check [here](https://github.com/yfeng95/SCARF/blob/main/process_data/README.md). 

## Getting Started
### Environment 
For face video, DELTA needs input image, subject mask, hair mask, and inital SMPL-X estimation for training. 
Specificly, we use
* [face alignment](https://github.com/1adrianb/face-alignment) to detect face keypoints
* [MediaPipe](hhttps://github.com/patlevin/face-detection-tflite) to detect iris
* [MODNet](https://github.com/ZHKKKe/MODNet) to segment subject
* [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch) to segment hair  

And model data from [DECA](https://github.com/yfeng95/DECA) and [PIXIE](https://github.com/yfeng95/PIXIE). 

When using the processing script, it is necessary to agree to the terms of their licenses and properly cite them in your work. 

1. Clone submodule repositories:
```
git submodule update --init --recursive
```
2. Download their needed data:
```bash
bash fetch_asset_data.sh
```
If the script failed, please check their websites and download the models manually. 

### process video data
Put your data list into ./lists/subject_list.txt, it can be video path or image folders.   
Using subject "7ka4tohxYD8_8" for examples: 

First process the video and generate labels  
```bash
cd process_data
python 0_process_video.py --videopath ../dataset/7ka4tohxYD8_8/7ka4tohxYD8_8.mp4 --savepath ../dataset --crop --ignore_existing
```
Then fit each image by optimizing SMPLX parameters, the process time depends on the number of frames, better to run each image parallelly if you have cluster :) 
```bash
cd ..
python process_data/1_smplx_fit_single.py --datapath dataset --subject 7ka4tohxYD8_8 --data_cfg dataset/7ka4tohxYD8_8/7ka4tohxYD8_8.yml --train_only
```
Then fit the video by optimizing all frames
```bash
python process_data/2_smplx_fit_all.py --datapath dataset --subject 7ka4tohxYD8_8 --data_cfg dataset/7ka4tohxYD8_8/7ka4tohxYD8_8.yml
```
You can check the fitting video results at `dataset/7ka4tohxYD8_8/smplx_all/epoch_000499.mp4`

The final results should look like:
<div align="center" dis>
    <table class="images" width="100%"  style="border:0px solid white; width:100%;">
        <tr style="border: 0px;">
            <td style="border: 0px;"><img src="../Doc/dataset/fit_person_0004.gif" width="256"></a></td>
            <td style="border: 0px;"><img src="../Doc/dataset/fit_MVI_1810.gif" width="256"></a></td>   
        </tr>
    </table>
</div>