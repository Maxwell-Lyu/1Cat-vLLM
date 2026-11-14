# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SD state-page transfer and fail-closed handshake tests, without a GPU engine."""

import ctypes
import unittest
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import msgspec
import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import NixlConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


class TestSDStatePages(unittest.TestCase):
    def make_worker(self, physical=1):
        worker = object.__new__(NixlConnectorWorker)
        worker._has_mamba = True
        worker._mamba_sd_same_tp = True
        worker._mamba_transfer_mode = "sd_same_tp_v1"
        worker.world_size = 4
        worker.block_size = 2048
        worker.block_len_per_layer = [256, 512]
        worker.kv_cache_layout = "HND"
        worker._mamba_ssm_size = (48, 128)
        worker._physical_blocks_per_logical_kv_block = physical
        worker._logical_num_blocks = 8
        worker.device_id = 0
        worker.num_regions = 4
        worker._group_spec_types = [FullAttentionSpec, MambaSpec]
        worker._remote_agents = defaultdict(dict)
        worker.nixl_wrapper = Mock()
        return worker

    @staticmethod
    def make_blocks(groups, cached=0):
        return SimpleNamespace(
            blocks=[
                [
                    SimpleNamespace(
                        block_id=value,
                        is_null=value == 0,
                        block_hash=b"cached" if index < cached else None,
                    )
                    for index, value in enumerate(group)
                ]
                for group in groups
            ]
        )

    def make_meta(self, worker, addresses=None):
        return NixlAgentMetadata(
            engine_id="P",
            agent_metadata=b"unused",
            kv_caches_base_addr=addresses or [1000, 10000],
            device_id=0,
            num_blocks=5 * worker._physical_blocks_per_logical_kv_block,
            block_lens=worker.block_len_per_layer,
            kv_cache_layout="HND",
            block_size=2048,
            ssm_sizes=worker._mamba_ssm_size,
            attn_backend_name="FLASH_ATTN_V100",
            physical_blocks_per_logical_kv_block=(
                worker._physical_blocks_per_logical_kv_block
            ),
            mamba_transfer_mode="sd_same_tp_v1",
        )

    def test_mixed_group_transfer_preserves_sd_states_and_padding(self):
        """Exercise descriptor selection with different P/D cache capacities."""
        for physical in (1, 2):
            with self.subTest(physical=physical):
                worker = self.make_worker(physical)
                sources = []
                targets = []
                for stride in worker.block_len_per_layer:
                    source = torch.arange(5 * stride * physical).to(torch.uint8)
                    sources.append(source.reshape(5, stride * physical))
                    targets.append(
                        torch.full((8, stride * physical), 239, dtype=torch.uint8)
                    )
                meta = self.make_meta(worker, [s.data_ptr() for s in sources])
                worker._validate_mamba_transfer_peer(meta, 4)
                local = worker._build_mamba_local([t.data_ptr() for t in targets], 1)
                remote = worker._build_mamba_remote(
                    meta,
                    1,
                    SimpleNamespace(remote_physical_blocks_per_logical=physical),
                )
                # FA descriptors precede the whole-page SSM descriptor section.
                local = [(0, 0, 0)] * (4 * 8 * physical) + local
                remote = [(0, 0, 0)] * (4 * 5 * physical) + remote
                dst_ids = worker._compute_desc_ids(
                    [[0], [2, 6]], 8 * physical, None, physical
                )[4:]
                src_ids = worker._compute_desc_ids(
                    [[0], [1, 3]], 5 * physical, None, physical
                )[4:]
                for dst, src in zip(dst_ids, src_ids):
                    dst_addr, dst_size, _ = local[dst]
                    src_addr, src_size, _ = remote[src]
                    self.assertEqual(dst_size, src_size)
                    ctypes.memmove(dst_addr, src_addr, src_size)
                for source, target in zip(sources, targets):
                    torch.testing.assert_close(target[2], source[1])
                    torch.testing.assert_close(target[6], source[3])
                    self.assertTrue(torch.all(target[[0, 1, 3, 4, 5, 7]] == 239))
                    # SD conv and temporal state values survive byte-for-byte.
                    for block, remote_block in ((2, 1), (6, 3)):
                        torch.testing.assert_close(
                            target[block, :48].view(torch.float16).reshape(3, 8),
                            source[remote_block, :48].view(torch.float16).reshape(3, 8),
                            equal_nan=True,
                        )
                        self.assertEqual(
                            target[block, 48:176].tolist(),
                            source[remote_block, 48:176].tolist(),
                        )

    def test_incompatible_peer_rejected_before_nixl_registration(self):
        cases: list[tuple[dict[str, Any], int]] = [
            ({"mamba_transfer_mode": "ds_split"}, 4),
            ({}, 2),
            ({"block_size": 1024}, 4),
            ({"ssm_sizes": (32, 128)}, 4),
            ({"block_lens": [512, 512]}, 4),
            ({"kv_caches_base_addr": [1000]}, 4),
            ({"kv_cache_layout": "NHD"}, 4),
            ({"physical_blocks_per_logical_kv_block": 2}, 4),
        ]
        for changes, tp in cases:
            with self.subTest(changes=changes, tp=tp):
                worker = self.make_worker()
                meta = replace(self.make_meta(worker), **changes)
                with self.assertRaises(ValueError):
                    worker.add_remote_agent(meta, remote_tp_size=tp)
                worker.nixl_wrapper.add_remote_agent.assert_not_called()

    def test_partial_logical_page_rejected(self):
        worker = self.make_worker(2)
        meta = replace(self.make_meta(worker), num_blocks=9)
        with self.assertRaisesRegex(ValueError, "partial logical block"):
            worker._validate_mamba_transfer_peer(meta, 4)

    def test_state_cannot_exceed_page(self):
        worker = self.make_worker()
        worker._mamba_ssm_size = (200, 128)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            worker._validate_mamba_transfer_peer(self.make_meta(worker), 4)

    def test_mixed_mode_rejected_even_when_hash_checks_are_disabled(self):
        worker = self.make_worker()
        worker._mamba_sd_same_tp = False
        worker._mamba_transfer_mode = "ds_split"
        with self.assertRaisesRegex(ValueError, "transfer modes"):
            worker._validate_mamba_transfer_peer(self.make_meta(worker), 4)

    def test_ds_projection_descriptors_unchanged(self):
        worker = self.make_worker()
        worker._mamba_sd_same_tp = False
        worker._logical_num_blocks = 2
        worker.block_len_per_layer = [256]
        worker._mamba_ssm_size = (48, 64)
        worker._conv_decomp = MambaConvSplitInfo(
            conv_rows=3,
            local_proj_dims=(2, 2, 4),
            conv_dtype_size=2,
            ssm_sizes=(48, 64),
        )
        self.assertEqual(
            worker._build_mamba_local([1000], 1),
            [
                (1000, 12, 0),
                (1256, 12, 0),
                (1012, 12, 0),
                (1268, 12, 0),
                (1024, 24, 0),
                (1280, 24, 0),
                (1048, 64, 0),
                (1304, 64, 0),
            ],
        )

    def test_align_tables_skip_cached_and_null_prefix_slots(self):
        worker = self.make_worker()
        local = [[20, 21], [40, 41, 42]]
        remote = [[10, 11, 12, 13], [0, 0, 30, 31, 32, 33]]
        aligned_local, aligned_remote = worker._apply_prefix_caching(local, remote, 1)
        self.assertEqual(aligned_local, local)
        self.assertEqual(aligned_remote, [[12, 13], [31, 32, 33]])
        self.assertEqual(remote[1], [0, 0, 30, 31, 32, 33])

    def test_fully_cached_groups_do_not_transfer_prefix_pages(self):
        worker = self.make_worker()
        self.assertEqual(
            worker._apply_prefix_caching([[], []], [[1, 2], [0, 3]], 1),
            ([[], []], [[], []]),
        )

    def test_missing_remote_suffix_rejected(self):
        worker = self.make_worker()
        with self.assertRaisesRegex(ValueError, "fewer blocks"):
            worker._apply_prefix_caching([[1, 2], [3]], [[4], [5]], 1)

    def test_empty_cached_groups_keep_integer_descriptor_ids(self):
        worker = self.make_worker()
        ids = worker._compute_desc_ids([[], [2]], 8, None, 1)
        np.testing.assert_array_equal(ids, [34, 42])
        self.assertEqual(ids.dtype, np.dtype(np.int64))
        ids = worker._compute_desc_ids([[], []], 8, None, 1)
        self.assertEqual(ids.size, 0)
        self.assertEqual(ids.dtype, np.dtype(np.int64))

    def test_attention_lookahead_pages_excluded_at_block_boundaries(self):
        scheduler = object.__new__(NixlConnectorScheduler)
        scheduler.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(
                        block_size=2048,
                        num_kv_heads=1,
                        head_size=128,
                        dtype=torch.float16,
                    )
                ),
            ]
        )
        for tokens, cached, local, expected_local, expected_remote in [
            (2046, 0, [20, 21], [20], [10]),
            (2047, 0, [20, 21], [20], [10]),
            (2048, 0, [20, 21], [20], [10]),
            (7230, 4096, [20, 21, 22], [20, 21], [12, 13]),
            (8190, 2048, [20, 21, 22, 23], [20, 21, 22], [11, 12, 13]),
        ]:
            with self.subTest(tokens=tokens, cached=cached):
                params = {
                    "remote_num_tokens": tokens,
                    "remote_block_ids": [[10, 11, 12, 13, 14]],
                }
                result = scheduler._clip_sd_prefill_blocks(
                    self.make_blocks(
                        [list(range(100, 100 + cached // 2048)) + local],
                        cached // 2048,
                    ),
                    params,
                    tokens - cached,
                )
                self.assertEqual(result, [expected_local])
                self.assertEqual(params["remote_block_ids"], [expected_remote])

    def test_uniform_attention_group_excludes_future_pages(self):
        scheduler = object.__new__(NixlConnectorScheduler)
        spec = FullAttentionSpec(
            block_size=2048, num_kv_heads=1, head_size=128, dtype=torch.float16
        )
        scheduler.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=UniformTypeKVCacheSpecs(
                        block_size=2048,
                        kv_cache_specs={"draft": spec},
                    )
                ),
            ]
        )
        params = {"remote_num_tokens": 2046, "remote_block_ids": [[10]]}
        self.assertEqual(
            scheduler._clip_sd_prefill_blocks(
                self.make_blocks([[20, 21]]), params, 2046
            ),
            [[20]],
        )
        self.assertEqual(params["remote_block_ids"], [[10]])

    def test_sliding_window_uses_absolute_positions(self):
        scheduler = object.__new__(NixlConnectorScheduler)
        scheduler.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SlidingWindowSpec(
                        block_size=2048,
                        num_kv_heads=1,
                        head_size=128,
                        dtype=torch.float16,
                        sliding_window=2048,
                    )
                ),
            ]
        )
        for tokens, local, remote, expected_local, expected_remote in [
            (2046, [20, 21], [10, 11], [20], [10]),
            (7230, [0, 0, 22, 23, 24], [0, 0, 12, 13, 14], [22, 23], [12, 13]),
        ]:
            params = {"remote_num_tokens": tokens, "remote_block_ids": [remote]}
            result = scheduler._clip_sd_prefill_blocks(
                self.make_blocks([local]),
                params,
                tokens,
            )
            self.assertEqual(result, [expected_local])
            self.assertEqual(params["remote_block_ids"], [expected_remote])
        blocks = self.make_blocks([[0, 0, 22, 23, 24]], cached=3)
        params = {"remote_num_tokens": 7230, "remote_block_ids": [[0, 0, 12, 13, 14]]}
        self.assertEqual(
            scheduler._clip_sd_prefill_blocks(blocks, params, 1086), [[23]]
        )
        self.assertEqual(params["remote_block_ids"], [[13]])

    def test_metadata_round_trip_includes_mode(self):
        worker = self.make_worker()
        meta = self.make_meta(worker)
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(meta), type=NixlAgentMetadata
        )
        self.assertEqual(decoded, meta)
        self.assertEqual(decoded.mamba_transfer_mode, "sd_same_tp_v1")


if __name__ == "__main__":
    unittest.main()
