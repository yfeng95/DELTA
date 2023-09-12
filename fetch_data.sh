#!/bin/bash
mkdir -p ./data

# SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
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
cd ./data
unzip utilities.zip
rm utilities.zip

# detla utilities
echo -e "\nDownloading delta data..."
wget https://nextcloud.tuebingen.mpg.de/index.php/s/zR3DM3zEdje984c/download -O ./data/delta_utilities.zip
unzip ./data/delta_utilities.zip -d ./data
mv ./data/delta_utilities/* ./data/
rm ./data/delta_utilities.zip
rm -rf ./data/delta_utilities

# # download two examples for visualization
mkdir -p ./exps
mkdir -p ./exps/released_version
cd exps/released_version
echo -e "\nDownloading DELTA trained avatars..."
wget https://nextcloud.tuebingen.mpg.de/index.php/s/KFzsWNgi4QJDqBa/download -O person_2_train.zip
unzip person_2_train.zip -d .
rm ./person_2_train.zip
wget https://nextcloud.tuebingen.mpg.de/index.php/s/feyzG7gAYxjkaHD/download -O person_0004.zip
unzip person_0004.zip -d .
rm ./person_0004.zip
cd ../..

# download 1 video example for data processing
mkdir -p ./dataset
cd ./dataset
wget https://nextcloud.tuebingen.mpg.de/index.php/s/BeQ4eYLLcAkNEBW/download -O 7ka4tohxYD8_8.zip
unzip 7ka4tohxYD8_8.zip -d .
rm ./7ka4tohxYD8_8.zip

## download processed data for training
# the original video data is from https://github.com/philgras/neural-head-avatars and https://github.com/gafniguy/4D-Facial-Avatars
# please cite their papers if you use the data
echo -e "\nDownloading DELTA processed data for training..."
wget https://nextcloud.tuebingen.mpg.de/index.php/s/ApBAtJXTck5gzFN/download -O person_0004.zip
unzip person_0004.zip -d .
rm ./person_0004.zip
wget https://nextcloud.tuebingen.mpg.de/index.php/s/E6a8Z4oP9SDStGb/download -O person_2_train.zip
unzip person_2_train.zip -d .
rm ./person_2_train.zip
cd ..