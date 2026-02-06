import time
import unittest

import numpy as np
from xpu.test_collective_api_base import TestDistBase

import paddle
import paddle.distributed as dist
import paddle.distributed.communication.deep_ep as ep
from paddle.base.core import Config
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group

import contextlib
import random
from functools import partial
import time

num_max_tokens = 128

# def bench_split(fn1, fn2, num_warmups: int = 50, num_tests: int = 50):
#     # clear
#     cache = paddle.empty((int(256e6 // 4),), dtype="int32")
#     cache.zero_()

#     # Warmup
#     for _ in range(num_warmups):
#         fn1()
#         fn2()

#     # Flush L2
#     cache.zero_()
#     del cache

#     # Testing
#     start_events_fn1 = [
#         paddle.device.Event(enable_timing=True) for _ in range(num_tests)
#     ]
#     end_events_fn1 = [
#         paddle.device.Event(enable_timing=True) for _ in range(num_tests)
#     ]
#     start_events_fn2 = [
#         paddle.device.Event(enable_timing=True) for _ in range(num_tests)
#     ]
#     end_events_fn2 = [
#         paddle.device.Event(enable_timing=True) for _ in range(num_tests)
#     ]
#     for i in range(num_tests):
#         # Record
#         start_events_fn1[i].record()
#         fn1()
#         end_events_fn1[i].record()
#         start_events_fn2[i].record()
#         fn2()
#         end_events_fn2[i].record()
#     paddle.device.synchronize()

#     times_fn1 = np.array(
#         [
#             s.elapsed_time(e) / 1e3
#             for s, e in zip(start_events_fn1, end_events_fn1)
#         ]
#     )[1:]
#     times_fn2 = np.array(
#         [
#             s.elapsed_time(e) / 1e3
#             for s, e in zip(start_events_fn2, end_events_fn2)
#         ]
#     )[1:]
#     return (
#         np.average(times_fn1),
#         np.min(times_fn1),
#         np.max(times_fn1),
#         np.average(times_fn2),
#         np.min(times_fn2),
#         np.max(times_fn2),
#     )


def bench_split(fn1, fn2, num_warmups=50, num_tests=50):
    # clear cache
    cache = paddle.empty((int(256e6 // 4),), dtype="int32")
    cache.zero_()

    # Warm up
    for _ in range(num_warmups):
        fn1()
        fn2()

    cache.zero_()
    del cache

    paddle.device.synchronize()

    # ---- XPU timing: use wall clock ----
    times_fn1 = []
    times_fn2 = []

    for _ in range(num_tests):
        # ---- fn1 ----
        t0 = time.time()
        fn1()
        paddle.device.synchronize()
        times_fn1.append((time.time() - t0) * 1e3)  # ms

        # ---- fn2 ----
        t0 = time.time()
        fn2()
        paddle.device.synchronize()
        times_fn2.append((time.time() - t0) * 1e3)  # ms

    t1 = np.array(times_fn1[1:])
    t2 = np.array(times_fn2[1:])

    return (
        np.mean(t1), np.min(t1), np.max(t1),
        np.mean(t2), np.min(t2), np.max(t2),
    )


# def bench(fn, num_warmups: int = 50, num_tests: int = 50):
#     # clear
#     cache = paddle.empty((int(256e6 // 4),), dtype="int32")
#     cache.zero_()

#     # Warmup
#     for _ in range(num_warmups):
#         fn()

#     # Flush L2
#     cache.zero_()
#     del cache

#     # Testing
#     start_events_fn = [
#         paddle.device.Event(enable_timing=True) for _ in range(num_tests)
#     ]
#     end_events_fn = [
#         paddle.device.Event(enable_timing=True) for _ in range(num_tests)
#     ]
#     for i in range(num_tests):
#         start_events_fn[i].record()
#         fn()
#         end_events_fn[i].record()
#     paddle.device.synchronize()

#     times_fn = np.array(
#         [
#             s.elapsed_time(e) / 1e3
#             for s, e in zip(start_events_fn, end_events_fn)
#         ]
#     )[1:]
#     return (
#         np.average(times_fn),
#         np.min(times_fn),
#         np.max(times_fn),
#     )

def bench(fn, num_warmups=50, num_tests=50):
    # warmup
    for _ in range(num_warmups):
        fn()
    paddle.device.synchronize()

    times = []
    for _ in range(num_tests):
        start = time.time()
        fn()
        paddle.device.synchronize()
        times.append((time.time() - start) * 1e3)  # ms
    return np.mean(times), np.min(times), np.max(times)

def per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
    x_fp32 = x_fp8.to("float32").view((x_fp8.shape[0], -1, 128))
    x_scales = x_scales.view((x_fp8.shape[0], -1, 1))
    return (x_fp32 * x_scales).view(x_fp8.shape).to("bfloat16")

def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    use_fp8: bool,
    rank: int,
    num_ranks: int,
    group: dist.communication.group,
    buffer: ep.Buffer,
    seed: int = 0,
):
    paddle.seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    num_rdma_ranks = num_ranks / 8

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, (
        'Too many ranks (exceeding test precision limit)'
    )

    x = paddle.ones((num_tokens, hidden), dtype="bfloat16") * (
        rank - rank_offset
    )
    x[:, -128:] = paddle.arange(0, num_tokens, dtype="bfloat16").view((-1, 1))
    topk_idx = paddle.randint(
        0, num_experts, shape=[num_tokens, num_topk], dtype="int64"
    )
    print(f"rank: {rank}, num_local_experts: {num_local_experts}")
    topk_weights = paddle.randn((num_tokens, num_topk), dtype="float32").abs_()
    # print("x: ", x, flush=True)
    # print("topk_idx: ", topk_idx, flush=True)
    # print("topk_weights: ", topk_weights, flush=True)

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    paddle.device.synchronize()
    dist.barrier()
    run_time = 100
    print("run_time: ", run_time)
    print("num_experts: ", num_experts)
    for return_recv_hook in (False, True):
        print(f"rank: {rank}, use_fp8: {use_fp8}", flush=True)
        for i in range(run_time):
            (
                packed_recv_x,
                packed_recv_count,
                handle,
                event,
                hook,
            ) = buffer.low_latency_dispatch(
                x,
                topk_idx,
                None,  # expertwise_scale
                num_max_tokens,
                num_experts,
                use_fp8=use_fp8,
                async_finish=not return_recv_hook,
                return_recv_hook=return_recv_hook,
            )
            hook() if return_recv_hook else event.current_stream_wait()
            packed_recv_x = (
                (packed_recv_x[0], packed_recv_x[1].contiguous())
                if use_fp8
                else packed_recv_x
            )
            # if use_fp8:
            #     print(
            #         "packed_recv_x: ", paddle.cast(packed_recv_x[0], "float32")
            #     )
            # else:
            #     print("packed_recv_x: ", paddle.cast(packed_recv_x, "float32"))
            out = paddle.empty((num_tokens, hidden), dtype="bfloat16")
            if use_fp8:
                simulated_gemm_x = per_token_cast_back(
                    packed_recv_x[0].view((-1, hidden)),
                    packed_recv_x[1].view((-1, hidden // 128)),
                ).view(packed_recv_x[0].shape)
            else:
                simulated_gemm_x = packed_recv_x.clone()

            # print("x:")
            # print(packed_recv_x)
            # print(simulated_gemm_x)
            # print("topk_idx:")
            # print(topk_idx)
            # print("topk_weights:")
            # print(topk_weights)
            # print("handle:")
            # print(handle)
            # return
            combined_x, event, hook = buffer.low_latency_combine(
                simulated_gemm_x,
                topk_idx,
                topk_weights,
                handle,
                async_finish=not return_recv_hook,
                zero_copy=False,
                return_recv_hook=return_recv_hook,
                out=out,
            )
            hook() if return_recv_hook else event.current_stream_wait()
            # print(f"rank: {rank}, combined_x: {combined_x}", flush=True)
            # return
        dist.barrier()
        paddle.device.synchronize()
        print("warmup_done", flush=True)

        def test_func(return_recv_hook: bool, do_send: bool, do_recv: bool):
            if do_send:
                (
                    packed_recv_x,
                    packed_recv_count,
                    _,
                    event,
                    hook,
                ) = buffer.low_latency_dispatch(
                    x,
                    topk_idx,
                    None,  # expertwise_scale
                    num_max_tokens,
                    num_experts,
                    use_fp8=use_fp8,
                    async_finish=not return_recv_hook,
                    return_recv_hook=return_recv_hook,
                )
                hook() if return_recv_hook else event.current_stream_wait()
            if do_recv:
                combined_x, event, hook = buffer.low_latency_combine(
                    simulated_gemm_x,
                    topk_idx,
                    topk_weights,
                    handle,
                    async_finish=not return_recv_hook,
                    zero_copy=False,
                    return_recv_hook=return_recv_hook,
                    out=out,
                )
                hook() if return_recv_hook else event.current_stream_wait()

        # dispatch + combine
        for do_send, do_recv in [
            (True, True),
        ]:
            avg_t_fn, min_t_fn, max_t_fn = bench(
                partial(
                    test_func,
                    return_recv_hook=False,
                    do_send=do_send,
                    do_recv=do_recv,
                ),  # combine
                num_warmups=50,
                num_tests=100,
            )
            if do_send and do_recv:
                print(
                    f'[rank {rank}] Dispatch + Combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / (avg_t_fn):.2f} GB/s, '
                    f'avg_t={(avg_t_fn) * 1e6:.2f} us, min_t={(min_t_fn) * 1e6:.2f} us, max_t={(max_t_fn) * 1e6:.2f} us',
                    flush=True,
                )
            elif do_send:
                print(
                    f'[rank {rank}] Dispatch bandwidth: {(num_dispatch_comm_bytes) / 1e9 / avg_t_fn:.2f} GB/s, '
                    f'avg_t={avg_t_fn * 1e6:.2f} us, min_t={min_t_fn * 1e6:.2f} us, max_t={max_t_fn * 1e6:.2f} us',
                    flush=True,
                )
            elif do_recv:
                print(
                    f'[rank {rank}] Combine bandwidth: {(num_combine_comm_bytes) / 1e9 / avg_t_fn:.2f} GB/s, '
                    f'avg_t={avg_t_fn * 1e6:.2f} us, min_t={min_t_fn * 1e6:.2f} us, max_t={max_t_fn * 1e6:.2f} us',
                    flush=True,
                )
            paddle.device.synchronize()
            dist.barrier()

        avg_t_fn1, min_t_fn1, max_t_fn1, avg_t_fn2, min_t_fn2, max_t_fn2 = (
            bench_split(
                partial(
                    test_func,
                    return_recv_hook=False,
                    do_send=True,
                    do_recv=False,
                ),  # dispatch
                partial(
                    test_func,
                    return_recv_hook=False,
                    do_send=False,
                    do_recv=True,
                ),  # combine
                num_warmups=50,
                num_tests=100,
            )
        )
        print(
            f'[rank {rank}] Dispatch bandwidth: {(num_dispatch_comm_bytes) / 1e9 / avg_t_fn1:.2f} GB/s, '
            f'avg_t={avg_t_fn1 * 1e6:.2f} us, min_t={min_t_fn1 * 1e6:.2f} us, max_t={max_t_fn1 * 1e6:.2f} us',
            flush=True,
        )
        print(
            f'[rank {rank}] Combine bandwidth: {(num_combine_comm_bytes) / 1e9 / avg_t_fn2:.2f} GB/s, '
            f'avg_t={avg_t_fn2 * 1e6:.2f} us, min_t={min_t_fn2 * 1e6:.2f} us, max_t={max_t_fn2 * 1e6:.2f} us',
            flush=True,
        )
        print(
            f'[rank {rank}] Dispatch + Combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / (avg_t_fn1 + avg_t_fn2):.2f} GB/s, '
            f'avg_t={(avg_t_fn1 + avg_t_fn2) * 1e6:.2f} us, min_t={(min_t_fn1 + min_t_fn2) * 1e6:.2f} us, max_t={(max_t_fn1 + max_t_fn2) * 1e6:.2f} us',
            flush=True,
        )


def test_loop():
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    group = paddle.distributed.new_group(range(num_ranks))
    print("rank: ", rank, flush=True)
    print("num_ranks: ", num_ranks, flush=True)

    num_tokens, hidden, num_topk, num_experts = 128, 7168, 8, 64
    assert num_tokens <= num_max_tokens, (
        "num_tokens must be less equal to num_max_tokens"
    )
    num_rdma_ranks = num_ranks / 8
    num_rdma_bytes = ep.Buffer.get_low_latency_rdma_size_hint(
        num_max_tokens, hidden, num_ranks, num_experts
    )
    num_nvl_bytes = 0
    use_fp8 = False
    print(
        f'Allocating rdma buffer size: {num_rdma_bytes / 1e6} MB, nvl buffer size: {num_nvl_bytes / 1e6} MB...',
        flush=True,
    )
    buffer = ep.Buffer(
        group,
        num_nvl_bytes=0,
        num_rdma_bytes=0,
        low_latency_mode=True,
        num_qps_per_rank=1,
    )
    test_main(
        num_tokens,
        hidden,
        num_experts,
        num_topk,
        use_fp8,
        rank,
        num_ranks,
        group,
        buffer,
        seed=1,
    )


def init_dist_env(world_size, seed=20):
    context = contextlib.nullcontext()
    with context:
        # start to init distributed env
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": world_size,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

        fleet.init(is_collective=True, strategy=strategy)


if __name__ == '__main__':
    if dist.get_world_size() > 1:
        init_dist_env(dist.get_world_size())
    test_loop()
