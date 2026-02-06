export PYTHONPATH=/home/Paddle/test/legacy_test:/home/Paddle/test:/home/Paddle/paddle/python:$PYTHONPATH
export LD_LIBRARY_PATH=/home/baidu/xpu/bkcl/output/so:$LD_LIBRARY_PATH

export BKCL_TIMEOUT=1000
export BKCL_SOCKET_IFNAME=eth0
# export BKCL_FORCE_RDMA_NICS_ORDER=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export BKCL_RDMA_NICS=eth1,eth2,eth3,eth4
export BKCL_ENABLE_XDR=1
# export XPULINK_VISIBLE_DEVICES=4,5,6,7
export XPU_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_DEVICE_ORDER=OAM_ID
# export BKCL_GID_INDEX=3
# export BKCL_DEEPEP_NORMAL_CLUSTER_NUM=8
# export BKCL_ENABLE_TREE=1
# export BKCL_TREE_THRESHOLD=4194304
# export BKCL_MULTI_TREE_THRESHOLD=4194304

# =========== xshmem ===========
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=64
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_USE_RDMA=1
export BKCL_FORCE_L3_RDMA=0   # 开1空间不够会报OOM错误
export BKCL_RDMA_VERBS=1

# =========== debug ===========
# export BKCL_C2C_CHECKSUM=2      # 1:单机  2:多机
# export BKCL_DIST_CHECK_INF_NAN=3   # 0b0001:单机input、output   0b00010:多机input、output    0b00100:多机跨机step
# export BKCL_FORCE_SYNC=1
# export DEEPEP_FORCE_SYNC=1
# export DEEPEP_DROP_AND_PAD=1

# export DEEPEP_SLEEP=1
# export BKCL_GROUP_DEBUG=1
# export BKCL_DUMP=31
# export BKCL_DEBUG=1 
# export XPUCUDA_DEBUG=1
# export XCCL_DBG_LEVEL=131072
# export XCCL_TESTS_SPLIT_MASK=1

# export GLOG_v=3
# export XPU_DEFAULT_STREAM_NUMBER=1
# export FLAGS_call_stack_level=2
# export XPURT_DISPATCH_MODE=PROFILING
# export XPU_PRINT_API_SUMMARY=1


# python -m paddle.distributed.launch --xpus $XPU_VISIBLE_DEVICES test/xpu/test_collective_deep_ep_alltoall_intranode.py
# python -m paddle.distributed.launch --xpus $XPU_VISIBLE_DEVICES --nnodes 2 --master "10.129.130.229:12345" test/xpu/test_collective_deepep_internode.py
python -m paddle.distributed.launch --xpus $XPU_VISIBLE_DEVICES test/xpu/test_collective_deepep_low_latency.py
# python -m paddle.distributed.launch --xpus $XPU_VISIBLE_DEVICES --nnodes 2 --master "10.129.130.229:12345" test/xpu/test_collective_deepep_low_latency.py
