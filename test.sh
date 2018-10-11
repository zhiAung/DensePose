export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /home/xiangyu.zhu/zhiang.hao/DensePose-master/
python2 tools/test_net.py \
    --cfg configs/Valmin_baseR50_e2e_dense.yaml \
    TEST.WEIGHTS detectron-output_50_36_dense/train/dense_coco_2014_valminusminival/generalized_rcnn/model_iter64999.pkl \
    NUM_GPUS 1
