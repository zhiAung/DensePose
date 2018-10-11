export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /home/xiangyu.zhu/zhiang.hao/DensePose-master/
#--cfg configs/valmin_cascade.yaml \
python2 tools/train_net.py \
    --cfg configs/Valmin_baseR50_e2e_dense.yaml \
    OUTPUT_DIR detectron-output_50_36_dense/
