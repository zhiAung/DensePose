cd /home/xiangyu.zhu/zhiang.hao/DensePose-master/
#--cfg configs/valmin_cascade.yaml \
python2 test/encode_results.py \
     detectron-output_coco_Experiment2/test/densepose_coco_2014_test/generalized_rcnn/body_uv_densepose_coco_2014_test_results.pkl  \
     detectron-output_coco_Experiment2/test/densepose_coco_2014_test/generalized_rcnn/densepose_test_res50_results.json
zip  -v \
densepose_test_res50_results.zip \
detectron-output_coco_Experiment2/test/densepose_coco_2014_test/generalized_rcnn/densepose_test_res50_results.json
