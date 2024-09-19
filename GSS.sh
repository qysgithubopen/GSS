#!/bin/bash
pythonname='pytorch_1.2.0a0+8554416-py36tf'

dataname='BRATS2020'
pypath=$pythonname
cudapath=cuda-9.0
datapath=${dataname}_Training_none_npy
savepath=output
 
export CUDA_VISIBLE_DEVICES=0,1

export PATH=$cudapath/bin:$PATH
export LD_LIBRARY_PATH=$cudapath/lib64:$LD_LIBRARY_PATH
PYTHON=$pypath/bin/python3.6
export PATH=$pypath/include:$pypath/bin:$PATH
export LD_LIBRARY_PATH=$pypath/lib:$LD_LIBRARY_PATH


savepath=output_2020_GSS_0
pretrain="./pretrain/model_last_brast2020.pth" #https://drive.google.com/file/d/1jK9KAaWfXXBpn3NlGBkn9NxrqSHu-rYG/view?pli=1
$PYTHON train_GSS.py  --batch_size=2  --datapath $datapath --savepath $savepath --num_epochs 300 --lr 2e-4 --region_fusion_start_epoch 20 --dataname $dataname --pretrain $pretrain

savepath=output_2020_GSS_1
pretrain="./output_2020_GSS_0/model_last.pth"
$PYTHON train_GSS.py  --batch_size=2  --datapath $datapath --savepath $savepath --num_epochs 300 --lr 2e-4 --region_fusion_start_epoch 20 --dataname $dataname --pretrain $pretrain

savepath=output_2020_GSS_2
pretrain="./output_2020_GSS_1/model_last.pth" 
$PYTHON train_GSS.py  --batch_size=2  --datapath $datapath --savepath $savepath --num_epochs 300 --lr 2e-4 --region_fusion_start_epoch 20 --dataname $dataname --pretrain $pretrain

savepath=output_2020_GSS_3
pretrain="./output_2020_GSS_3/model_last.pth" 
$PYTHON train_GSS.py  --batch_size=2  --datapath $datapath --savepath $savepath --num_epochs 300 --lr 2e-4 --region_fusion_start_epoch 20 --dataname $dataname --pretrain $pretrain

savepath=output_2020_GSS_4
pretrain="./output_2020_GSS_4/model_last.pth" 
$PYTHON train_GSS.py  --batch_size=2  --datapath $datapath --savepath $savepath --num_epochs 300 --lr 2e-4 --region_fusion_start_epoch 20 --dataname $dataname --pretrain $pretrain
