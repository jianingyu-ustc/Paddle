# export {http,https}_proxy=http://agent.baidu.com:8188
# export {http,https}_proxy=http://10.162.37.16:8128
# export {http,https}_proxy=http://gzbh-aip-paddlecloud140.gzbh:8128
# export {http,https}_proxy=http://172.19.57.45:3128
export {http,https}_proxy=http://10.63.229.53:8891
# declare -x {http,https}_proxy="http://gzbh-aip-paddlecloud140.gzbh:8128"

cd build

# cmake .. -DPY_VERSION=3.10 \
# -DCMAKE_BUILD_TYPE=Release \
# -DWITH_GPU=OFF \
# -DWITH_XPU=ON \
# -DON_INFER=OFF \
# -DWITH_PYTHON=ON \
# -DWITH_XPU_XRE5=ON \
# -DWITH_MKL=OFF \
# -DWITH_XPU_BKCL=ON \
# -DWITH_TESTING=ON \
# -DWITH_XCCL_RDMA=ON \
# -DWITH_XPU_XHPC=ON \
# -DBUILD_WHL_PACKAGE=ON \
# -DWITH_DISTRIBUTE=ON \
# -DARCH_BIN_CONTAINS_90=1

make -j$(nproc) TARGET=HASWELL

# pip install build/python/dist/ -I --force-reinstall
# pip install build/python/dist/ -I --force-reinstall --no-deps
