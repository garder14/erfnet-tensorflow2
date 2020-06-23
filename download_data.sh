#!/bin/bash

mkdir -p ./data &&

wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$1&&password=$2&&submit=Login" https://www.cityscapes-dataset.com/login/ &&

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 &&
unzip gtFine_trainvaltest.zip &&
rm gtFine_trainvaltest.zip license.txt README &&
mv ./gtFine ./data/segmentations &&

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 &&
unzip leftImg8bit_trainvaltest.zip &&
rm leftImg8bit_trainvaltest.zip license.txt README &&
mv ./leftImg8bit ./data/images &&

rm cookies.txt index.html
