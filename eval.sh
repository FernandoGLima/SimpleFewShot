#!/bin/bash


### Usage 
### nohup ./eval.sh &> output.out &


run_dir="./grid_search"
mkdir -p "$run_dir"

python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/dn4_2_5_lr0.001_l2_0.log" 2>&1 && \


python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model feat --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/feat_2_5_lr0.001_l2_0.log" 2>&1 && \


python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/matching_net_2_5_lr0.001_l2_0.log" 2>&1 && \

python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/metaopt_net_2_5_lr0.001_l2_0.log" 2>&1 && \

python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/msenet_2_5_lr0.001_l2_0.log" 2>&1 && \

python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/proto_net_2_5_lr0.001_l2_0.log" 2>&1 && \

python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/relation_net_2_5_lr0.001_l2_0.log" 2>&1 && \

python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/tadam_2_5_lr0.001_l2_0.log" 2>&1 && \

python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.001 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.001 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.001 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.001 --l2_weight 0.0001 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.0001 --l2_weight 0.0001 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 && \
python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 --lr 0.00001 --l2_weight 0.0001 >> "${run_dir}/tapnet_2_5_lr0.001_l2_0.log" 2>&1 



##### all
# python3 train.py --model dn4 --ways 2 --shots 5 --gpu 1 >> "${run_dir}/dn4_2_5.log" 2>&1 && \
# python3 train.py --model dn4 --ways 2 --shots 10 --gpu 1 >> ${run_dir}/dn4_2_10.log 2>&1 && \
# python3 train.py --model dn4 --ways 2 --shots 20 --gpu 1 >> ${run_dir}/dn4_2_20.log 2>&1 && \
# python3 train.py --model dn4 --ways 3 --shots 5 --gpu 1 >> ${run_dir}/dn4_3_5.log 2>&1 && \
# python3 train.py --model dn4 --ways 3 --shots 10 --gpu 1 >> ${run_dir}/dn4_3_10.log 2>&1 && \
# python3 train.py --model dn4 --ways 3 --shots 20 --gpu 1 >> ${run_dir}/dn4_3_20.log 2>&1 && \

# python3 train.py --model feat --ways 2 --shots 5 --gpu 1 >> ${run_dir}/feat_2_5.log 2>&1 && \
# python3 train.py --model feat --ways 2 --shots 10 --gpu 1 >> ${run_dir}/feat_2_10.log 2>&1 && \
# python3 train.py --model feat --ways 2 --shots 20 --gpu 1 >> ${run_dir}/feat_2_20.log 2>&1 && \
# python3 train.py --model feat --ways 3 --shots 5 --gpu 1 >> ${run_dir}/feat_3_5.log 2>&1 && \
# python3 train.py --model feat --ways 3 --shots 10 --gpu 1 >> ${run_dir}/feat_3_10.log 2>&1 && \
# python3 train.py --model feat --ways 3 --shots 20 --gpu 1 >> ${run_dir}/feat_3_20.log 2>&1 && \

# python3 train.py --model matching_net --ways 2 --shots 5 --gpu 1 >> ${run_dir}/matching_net_2_5.log 2>&1 && \
# python3 train.py --model matching_net --ways 2 --shots 10 --gpu 1 >> ${run_dir}/matching_net_2_10.log 2>&1 && \
# python3 train.py --model matching_net --ways 2 --shots 20 --gpu 1 >> ${run_dir}/matching_net_2_20.log 2>&1 && \
# python3 train.py --model matching_net --ways 3 --shots 5 --gpu 1 >> ${run_dir}/matching_net_3_5.log 2>&1 && \
# python3 train.py --model matching_net --ways 3 --shots 10 --gpu 1 >> ${run_dir}/matching_net_3_10.log 2>&1 && \
# python3 train.py --model matching_net --ways 3 --shots 20 --gpu 1 >> ${run_dir}/matching_net_3_20.log 2>&1 && \

# python3 train.py --model metaopt_net --ways 2 --shots 5 --gpu 0 >> ${run_dir}/metaopt_net_2_5.log 2>&1 && \
# python3 train.py --model metaopt_net --ways 2 --shots 10 --gpu 0 >> ${run_dir}/metaopt_net_2_10.log 2>&1 && \
# python3 train.py --model metaopt_net --ways 2 --shots 20 --gpu 0 >> ${run_dir}/metaopt_net_2_20.log 2>&1 && \
# python3 train.py --model metaopt_net --ways 3 --shots 5 --gpu 0 >> ${run_dir}/metaopt_net_3_5.log 2>&1 && \
# python3 train.py --model metaopt_net --ways 3 --shots 10 --gpu 0 >> ${run_dir}/metaopt_net_3_10.log 2>&1 && \
# python3 train.py --model metaopt_net --ways 3 --shots 20 --gpu 0 >> ${run_dir}/metaopt_net_3_20.log 2>&1 && \

# python3 train.py --model msenet --ways 2 --shots 5 --gpu 1 >> ${run_dir}/msenet_2_5.log 2>&1 && \
# python3 train.py --model msenet --ways 2 --shots 10 --gpu 1 >> ${run_dir}/msenet_2_10.log 2>&1 && \
# python3 train.py --model msenet --ways 2 --shots 20 --gpu 1 >> ${run_dir}/msenet_2_20.log 2>&1 && \
# python3 train.py --model msenet --ways 3 --shots 5 --gpu 1 >> ${run_dir}/msenet_3_5.log 2>&1 && \
# python3 train.py --model msenet --ways 3 --shots 10 --gpu 1 >> ${run_dir}/msenet_3_10.log 2>&1 && \
# python3 train.py --model msenet --ways 3 --shots 20 --gpu 1 >> ${run_dir}/msenet_3_20.log 2>&1 && \

# python3 train.py --model proto_net --ways 2 --shots 5 --gpu 1 >> ${run_dir}/proto_net_2_5.log 2>&1 && \
# python3 train.py --model proto_net --ways 2 --shots 10 --gpu 1 >> ${run_dir}/proto_net_2_10.log 2>&1 && \
# python3 train.py --model proto_net --ways 2 --shots 20 --gpu 1 >> ${run_dir}/proto_net_2_20.log 2>&1 && \
# python3 train.py --model proto_net --ways 3 --shots 5 --gpu 1 >> ${run_dir}/proto_net_3_5.log 2>&1 && \
# python3 train.py --model proto_net --ways 3 --shots 10 --gpu 1 >> ${run_dir}/proto_net_3_10.log 2>&1 && \
# python3 train.py --model proto_net --ways 3 --shots 20 --gpu 1 >> ${run_dir}/proto_net_3_20.log 2>&1 && \

# python3 train.py --model relation_net --ways 2 --shots 5 --gpu 1 >> ${run_dir}/relation_net_2_5.log 2>&1 && \
# python3 train.py --model relation_net --ways 2 --shots 10 --gpu 1 >> ${run_dir}/relation_net_2_10.log 2>&1 && \
# python3 train.py --model relation_net --ways 2 --shots 20 --gpu 1 >> ${run_dir}/relation_net_2_20.log 2>&1 && \
# python3 train.py --model relation_net --ways 3 --shots 5 --gpu 1 >> ${run_dir}/relation_net_3_5.log 2>&1 && \
# python3 train.py --model relation_net --ways 3 --shots 10 --gpu 1 >> ${run_dir}/relation_net_3_10.log 2>&1 && \
# python3 train.py --model relation_net --ways 3 --shots 20 --gpu 1 >> ${run_dir}/relation_net_3_20.log 2>&1 && \

# python3 train.py --model tadam --ways 2 --shots 5 --gpu 1 >> ${run_dir}/tadam_2_5.log 2>&1 && \
# python3 train.py --model tadam --ways 2 --shots 10 --gpu 1 >> ${run_dir}/tadam_2_10.log 2>&1 && \
# python3 train.py --model tadam --ways 2 --shots 20 --gpu 1 >> ${run_dir}/tadam_2_20.log 2>&1 && \
# python3 train.py --model tadam --ways 3 --shots 5 --gpu 1 >> ${run_dir}/tadam_3_5.log 2>&1 && \
# python3 train.py --model tadam --ways 3 --shots 10 --gpu 1 >> ${run_dir}/tadam_3_10.log 2>&1 && \
# python3 train.py --model tadam --ways 3 --shots 20 --gpu 1 >> ${run_dir}/tadam_3_20.log 2>&1 && \

# python3 train.py --model tapnet --ways 2 --shots 5 --gpu 1 >> ${run_dir}/tapnet_2_5.log 2>&1
# python3 train.py --model tapnet --ways 2 --shots 10 --gpu 1 >> ${run_dir}/tapnet_2_10.log 2>&1 && \
# python3 train.py --model tapnet --ways 2 --shots 20 --gpu 1 >> ${run_dir}/tapnet_2_20.log 2>&1 && \
# python3 train.py --model tapnet --ways 3 --shots 5 --gpu 1 >> ${run_dir}/tapnet_3_5.log 2>&1 && \
# python3 train.py --model tapnet --ways 3 --shots 10 --gpu 1 >> ${run_dir}/tapnet_3_10.log 2>&1 && \
# python3 train.py --model tapnet --ways 3 --shots 20 --gpu 1 >> ${run_dir}/tapnet_3_20.log 2>&1 
