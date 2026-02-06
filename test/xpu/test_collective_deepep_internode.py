# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import unittest
from typing import Any

import numpy as np
from xpu.test_collective_api_base import TestDistBase

import paddle
import paddle.distributed as dist
import paddle.distributed.communication.deep_ep as ep
from paddle.base.core import Config
from paddle.distributed import fleet


def get_num_local_ranks():
    visible = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv(
        "XPU_VISIBLE_DEVICES"
    )
    if visible is None:
        return 1
    return len(visible.split(","))


def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def align_up(x, y):
    return (x + y - 1) // y * y


def inplace_unique(x: paddle.Tensor, num_slots) -> paddle.Tensor:
    num_slots = int(num_slots)
    x_np = x.numpy().astype(int)  # ✅ 强制整数类型
    B, S = x_np.shape

    for i in range(B):
        row = x_np[i]

        counts = np.zeros(num_slots, dtype=int)
        for r in row:
            if 0 <= r < num_slots:  # 跳过 -1 或无效值
                counts[r] += 1

        sorted_idx = np.argsort(-counts)
        sorted_idx = [r for r in sorted_idx if counts[r] > 0]

        new_row = np.full(S, -1, dtype=int)
        for j, r in enumerate(sorted_idx):
            if j >= S:
                break
            new_row[j] = r
        x_np[i] = new_row

    return paddle.to_tensor(x_np, dtype=x.dtype)


def create_grouped_scores(
    scores: paddle.Tensor, group_idx: paddle.Tensor, num_groups: int
):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = paddle.zeros(
        (num_tokens, num_groups), dtype=paddle.bool, device=scores.device
    )
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def hash_tensor(t: paddle.Tensor):
    return t.view(paddle.int).sum().item()


# noinspection PyShadowingNames
def test_main(
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    buffer: ep.Buffer,
    group: Any,
    skip_benchmark: bool = False,
):
    # Settings
    # num_tokens, hidden = args.num_tokens, args.hidden
    # num_topk_groups, num_topk, num_experts = args.num_topk_groups, args.num_topk, args.num_experts
    num_tokens, hidden = 1024, 7168
    num_topk_groups, num_topk, num_experts = 1, 8, 256

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(
            f'{num_sms}, {local_rank}, {num_local_ranks}, {num_ranks}, {num_nodes}, {rank}, [config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}',
            flush=True,
        )

    # Random data
    x = paddle.ones((num_tokens, hidden), dtype=paddle.bfloat16) * rank
    x_pure_rand = paddle.randn((num_tokens, hidden), dtype=paddle.bfloat16)
    # x_e4m3 = per_token_cast_to_fp8(x)
    # x_pure_rand_e4m3 = per_token_cast_to_fp8(x_pure_rand)
    # x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    # scores = paddle.randn((num_tokens, num_experts), dtype=paddle.float32).abs() + 1
    # group_scores = scores.view([num_tokens, num_nodes, -1]).amax(axis=-1)
    # num_features_per_node = scores.shape[1] // num_nodes  # 整除，返回 int
    # new_shape = [num_tokens, num_nodes, num_features_per_node]
    # group_scores = scores.reshape(new_shape).amax(axis=-1)
    # group_idx = paddle.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    # masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    # topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    # topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    topk_idx = np.zeros([num_tokens, num_topk], dtype='int32')
    for i in range(num_tokens):
        topk_idx[i] = np.random.choice(
            np.arange(num_experts), [num_topk], replace=False
        )
    topk_idx = paddle.to_tensor(topk_idx)
    topk_weights = (
        paddle.ones((num_tokens, num_topk), dtype=paddle.float32) * rank
    )
    topk_weights_pure_rand = paddle.randn(
        (num_tokens, num_topk), dtype=paddle.float32
    )

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(paddle.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)
    hash_value = 0

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = (
        rdma_idx.ne(paddle.full_like(rdma_idx, -1)).sum().item()
    )

    # Expert meta
    num_tokens_per_expert = paddle.zeros((num_experts,), dtype=paddle.int32)
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = paddle.empty((num_ranks,), dtype=paddle.int32)
    num_tokens_per_rdma_rank = paddle.empty((num_nodes,), dtype=paddle.int32)
    token_idx_in_rank = paddle.full(
        (num_ranks, num_tokens), -1, dtype=paddle.int32
    )
    for i in range(num_ranks):
        mask = paddle.cast(rank_idx == i, dtype=paddle.int32)
        num_tokens_per_rank[i] = mask.sum()
        token_sel = paddle.max(mask, axis=-1)
        count = int(token_sel.sum().item())
        tokens = paddle.argsort(token_sel, axis=0, descending=True)
        if count > 0:
            top_tokens = tokens[:count]
            top_tokens = paddle.sort(top_tokens, axis=0)
            token_idx_in_rank[i][top_tokens] = paddle.arange(
                count, dtype=paddle.int32
            )

    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(paddle.int32)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    (
        ref_num_tokens_per_rank,
        ref_num_tokens_per_rdma_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)
    # assert paddle.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    # assert paddle.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    # assert paddle.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    # assert paddle.allclose(ref_is_token_in_rank, is_token_in_rank)
    # t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    # if local_rank == 0:
    #     print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
    #     print('', flush=True)
    dist.barrier(group=group)

    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = (
        128,
        (720 if num_ranks in (24, 48, 96, 144, 160) else 512),
    )
    config = Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert paddle.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (
                check_x[check_start:check_end, :].int() - i
            ).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (
                x_pure_rand,
                x,
            ):
                for with_topk in (False, True):
                    is_rand = current_x is x_pure_rand
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end='',
                        )
                    dispatch_args = {
                        'x': current_x,
                        'num_tokens_per_rank': num_tokens_per_rank,
                        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                        'is_token_in_rank': is_token_in_rank,
                        'num_tokens_per_expert': num_tokens_per_expert,
                        'config': config,
                        'async_finish': async_mode,
                    }
                    # if with_topk:   # TODO: Currently, cache mode needs with_topk=True when called for the first time
                    dispatch_args.update(
                        {
                            'topk_idx': topk_idx,
                            'topk_weights': topk_weights_pure_rand
                            if is_rand
                            else topk_weights,
                        }
                    )
                    if previous_mode:
                        dispatch_args.update(
                            {'previous_event': buffer.capture()}
                        )
                    (
                        recv_x,
                        recv_topk_idx,
                        recv_topk_weights,
                        recv_num_tokens_per_expert_list,
                        handle,
                        event,
                    ) = buffer.dispatch(**dispatch_args)

                    event.current_stream_wait() if async_mode else ()

                    if current_x is x_pure_rand or current_x is x:
                        hash_value += hash_tensor(recv_x)
                    else:
                        hash_value += hash_tensor(recv_x[0])
                        hash_value += hash_tensor(recv_x[1])

                    # recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]

                    # print("topk_idx:")
                    # print(topk_idx)
                    # print("num_tokens_per_rank:")
                    # print(num_tokens_per_rank)
                    # print("gbl_num_tokens_per_rank:")
                    # print(gbl_num_tokens_per_rank)
                    # print("recv_x:")
                    # print(recv_x)

                    # assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), \
                    #     f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    # assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    # if not is_rand:
                    # check_data(recv_x, recv_gbl_rank_prefix_sum)
                    # recv_topk_weights_clone = None
                    # if with_topk:
                    #     # Check `topk_idx`
                    #     assert (recv_topk_idx.eq(-1) |
                    #             ((recv_topk_idx >= 0) &
                    #              (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                    #     for i, count in enumerate(recv_num_tokens_per_expert_list):
                    #         assert recv_topk_idx.eq(i).sum().item() == count

                    #     # Check `topk_weights`
                    #     recv_topk_weights_clone = recv_topk_weights.clone()
                    #     if not is_rand:
                    #         recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(
                    #             dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                    #         check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test `num_worst_tokens != 0`
                    # if with_topk:
                    #     num_worst_tokens = num_tokens * num_ranks
                    #     dispatch_args.update({'num_worst_tokens': num_worst_tokens})
                    #     recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)
                    #     event.current_stream_wait() if async_mode else ()
                    #     recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x
                    #     assert len(empty_list) == 0
                    #     assert num_worst_tokens == recv_worst_x.size(0)
                    #     assert num_worst_tokens == recv_worst_topk_idx.size(0)
                    #     assert num_worst_tokens == recv_worst_topk_weights.size(0)
                    #     assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)])
                    #     assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:recv_x.size(0)])
                    #     assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:recv_x.size(0)])
                    #     assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {
                            'x': current_x,
                            'handle': handle,
                            'config': config,
                            'async_finish': async_mode,
                        }
                        if previous_mode:
                            dispatch_args.update(
                                {'previous_event': buffer.capture()}
                            )
                        recv_x, _, _, _, _, event = buffer.dispatch(
                            **dispatch_args
                        )
                        event.current_stream_wait() if async_mode else ()
                        # recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        # if not is_rand:
                        #     check_data(recv_x, recv_gbl_rank_prefix_sum)

                    # Test combine
                    # bias_0 = paddle.ones((num_tokens, hidden), dtype=paddle.bfloat16)
                    # bias_1 = paddle.randn((num_tokens, hidden), dtype=paddle.bfloat16)
                    combine_args = {
                        'x': recv_x,
                        'handle': handle,
                        'config': config,
                        'async_finish': async_mode,
                    }
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update(
                            {'previous_event': buffer.capture()}
                        )
                    combined_x, combined_topk_weights, event = buffer.combine(
                        **combine_args
                    )
                    event.current_stream_wait() if async_mode else ()
                    # check_x = (combined_x.float() - bias_0.float() - bias_1.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    # ref_x = x_pure_rand if is_rand else x
                    # assert calc_diff(check_x, ref_x) < 5e-4 if current_x is x_pure_rand_e4m3 else 5e-6
                    # if with_topk:
                    #     check_topk_weights = combined_topk_weights if is_rand else (combined_topk_weights /
                    #                                                                 is_token_in_rank.sum(dim=1).unsqueeze(1))
                    #     ref_topk_weights = topk_weights_pure_rand if is_rand else topk_weights
                    #     assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # hash_value += hash_tensor(recv_x)

                    # # For later tuning
                    # dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    # dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    # combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    # combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

    if local_rank == 0:
        print(flush=True)

    return hash_value


class TestCollectiveDeepEPAllToAllIntranode(TestDistBase):
    def test_loop(self):
        num_ranks = dist.get_world_size()
        rank_id = dist.get_rank()
        num_local_ranks = get_num_local_ranks()
        local_rank_id = rank_id % num_local_ranks
        num_nodes = num_ranks // num_local_ranks

        print(num_ranks, rank_id, num_local_ranks, num_nodes)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": num_ranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        seed = 20
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=strategy)
        ep_group = paddle.distributed.new_group(range(num_ranks))
        paddle.seed(rank_id)
        num_sms = 24
        print("ep_group.rank: ", ep_group.rank)

        test_ll_compatibility = False
        deepep_engine = ep.Buffer(
            ep_group,
            int(2e9),
            int(1e9),
            low_latency_mode=test_ll_compatibility,
            num_qps_per_rank=1,  # set in bkcl
        )
        assert num_local_ranks == 8 and num_ranks > 8

        # for seed in range(int(1e9)):
        for seed in range(25):
            if local_rank_id == 0:
                print(f'Testing with seed {seed} ...', flush=True)
            paddle.manual_seed(rank_id + seed)
            ref_hash = 0
            for i in (num_sms,):
                ref_hash += test_main(
                    i,
                    local_rank_id,
                    num_local_ranks,
                    num_ranks,
                    num_nodes,
                    rank_id,
                    deepep_engine,
                    ep_group,
                    0,
                )
                if local_rank_id == 0:
                    print(flush=True)


if __name__ == "__main__":
    unittest.main()
