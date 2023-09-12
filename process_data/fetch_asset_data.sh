# trained model for RobustVideoMatting
mkdir -p ./assets
mkdir -p ./assets/MODNet
echo -e "Downloading MODNet model..."
# https://drive.google.com/file/d/1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX/viewmkdir 
echo -e "Downloading MODNet model..."
FILEID=1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX
FILENAME=./assets/MODNet/modnet_webcam_portrait_matting.ckpt.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

# trained model for face-parsing
# if failed, please download the model from
# https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view 
echo -e "Downloading face-parsing model..."
FILEID=154JgKpzCPW82qINcVieuPH3fZ2e0P812
FILENAME=./assets/face_parsing/model.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

# trained model for PIXIE
# if failed, please check https://github.com/yfeng95/PIXIE/blob/master/fetch_model.sh
cd submodules/PIXIE
echo -e "Downloading PIXIE data..."
#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# PIXIE pretrained model and utilities
echo -e "\nYou need to register at https://pixie.is.tue.mpg.de/"
read -p "Username (PIXIE):" username
read -p "Password (PIXIE):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O './data/pixie_model.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=utilities.zip&resume=1' -O './data/utilities.zip' --no-check-certificate --continue

cd ..
echo -e "\nDownloading more data..."
wget https://nextcloud.tuebingen.mpg.de/index.php/s/jraekdRrxCzYEWB/download -O ./data/delta_utilities2.zip
unzip ./data/delta_utilities2.zip -d ./data
mv ./data/delta_utilities2/* ./data/
rm ./data/delta_utilities2.zip
rm -rf ./data/delta_utilities2
cd ../process_data

