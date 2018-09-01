## tricks

•	综述：复现Dense Human Pose Estimation In The Wild 论文的github项目，并参加了2018 PoseTrack DensePose Challenge，取得第一的名次。

•	实验：1、更改了源码的简单直筒型densepose_head，尝试使用更复杂的残差结构，及Hourglass结构,在部分实验训练集上有2%左右的提升。

•	实验：2、尝试多任务训练，利用多任务之间对网络的互补优势，在增加了keypoint task之后，准确率提升了3%左右 ，后面继续增加mask task之后，准确率有提升，但是不太明显。

•	实验：3、尝试测试增强,采用多尺度bbox输出取平均，准确率提升2%。

•	实验：4、在COCO Dataset上训练完之后，在PoseTrack Dataset上finetune，提升了1%的准确率。
