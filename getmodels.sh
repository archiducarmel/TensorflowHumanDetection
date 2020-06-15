#!/bin/sh -x #!/bin/bash

echo "###########################################################################################"
echo "###########################################################################################"
echo "#################                                                ##########################"
echo "#################     Start downloading ssd_resnet50 model       ##########################"
echo "#################                                                ##########################"
echo "###########################################################################################"
echo "###########################################################################################"
wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar -xzvf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
mv ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 object_detection/models
rm ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

echo "###########################################################################################"
echo "###########################################################################################"
echo "#################                                                ##########################"
echo "################# Start downloading faster_rcnn_resnet50 model   ##########################"
echo "#################                                                ##########################"
echo "###########################################################################################"
echo "###########################################################################################"
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
tar -xzvf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
mv faster_rcnn_resnet50_coco_2018_01_28 object_detection/models
rm faster_rcnn_resnet50_coco_2018_01_28.tar.gz

echo "###########################################################################################"
echo "###########################################################################################"
echo "#################                                                ##########################"
echo "#################    Start downloading ssd_mobilenet_v2 model    ##########################"
echo "#################                                                ##########################"
echo "###########################################################################################"
echo "###########################################################################################"
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
mv ssd_mobilenet_v2_coco_2018_03_29 object_detection/models
rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
#echo "Start downloading ssd_mobilenet_v2 model"
#wget -O object_detection/models/ssd_mobilenet_v2_coco_2018_03_29.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
#unzip object_detection/models/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
