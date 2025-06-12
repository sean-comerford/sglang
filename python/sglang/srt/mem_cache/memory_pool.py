"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
import KVUtil
import ctypes
from collections import deque
import time
"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import logging
import threading
from enum import IntEnum
from functools import wraps
from typing import List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import triton
import triton.language as tl

import os
import queue

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import debug_timing, get_compiler_backend

logger = logging.getLogger(__name__)


kv_logger = logging.getLogger("kv_logger")
kv_logger.setLevel(logging.INFO)

kv_logger.propagate = False


if not kv_logger.handlers:
    runtime_id = os.getenv("RUNTIME_ID")
    if runtime_id is None:
        raise RuntimeError("runtime_id is not set")

    log_filename = f"kv_cache_{runtime_id}.log"
    log_path = os.path.abspath(log_filename)

    kv_handler = logging.FileHandler(log_path)
    kv_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    kv_logger.addHandler(kv_handler)

    print(f"[kv_logger] Logging to: {log_path}")
    
    
GB = 1024 * 1024 * 1024
MB = 1024 * 1024


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data(self, indices):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer_per_layer(self, indices, flat_data, layer_id):
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter



class TokenToKVPoolAllocator:
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = 1

        self._kvcache = kvcache
        self.not_available = False
        
        self.clear()
        self.token_size = self._kvcache.head_num * self._kvcache.head_dim
        self.distance_layer = self._kvcache.head_num * self._kvcache.head_dim * (self.size + self.page_size)
      
    def shutdown(self):
        self._kvcache.close()
        
    def log_allocated_size(self):
        free_size = len(self.free_slots)
        allocated_size_indices = self.size + 1 - free_size
        size_in_bytes = 2 #float16
        allocated_size_layer_bytes = allocated_size_indices * self._kvcache.layer_num * self._kvcache.head_num * self._kvcache.head_dim * size_in_bytes
        kv_logger.info(
            f"Allocated KV Cache size: {allocated_size_layer_bytes * 2 / MB:.2f} MB"
        )
        
    def available_size(self):
        return len(self.free_slots)

    def get_kvcache(self):
        return self._kvcache
    
    def alloc(self, need_size: int):
        if need_size > len(self.free_slots) or self.not_available:
            return None
        
        start_time = time.perf_counter()
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        
        self._kvcache.alloc_job_queue.put(select_index)
        
        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1e6
        # kv_logger.info(
        #     f"alloc1 took {elapsed_us:.2f} µs"
        # )
        self.log_allocated_size()

        return select_index


    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []  
         
            
    def free(self, free_index):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            start_time = time.perf_counter()
            self.free_slots = torch.cat((self.free_slots, free_index))
                
            if isinstance(free_index, torch.Tensor):
                indices = free_index.cpu().tolist()
            else:
                indices = list(free_index)
                
            self._kvcache.kv_pool.mem_free(indices, self.token_size, self._kvcache.layer_num, self.distance_layer)
            end_time = time.perf_counter()
            elapsed_us = (end_time - start_time) * 1e6
            # kv_logger.info(
            #     f"free took {elapsed_us:.2f} µs"
            # )
            self.log_allocated_size()
        else: # tensor
            self.free_group.append(free_index)
     

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))
            self.free_group = []  
         
    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []

class MHATokenToKVPool(KVCache):
        
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        tp_size: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size

        self.dtype = dtype
        self.device = device
        self.store_dtype = dtype
        
        assert(dtype == torch.float16)
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=False
        )
        
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.kv_pool = KVUtil.KVAllocator("mem_access_time")
        self.tp_size = tp_size
        self._create_buffers()
        self.k_buffer_cache = {}
        self.v_buffer_cache = {}
        
        self.token_size = self.head_num * self.head_dim
        self.distance_layer = self.head_num * self.head_dim * (self.size + self.page_size)
      
        self.capture_mode = False
        self.device_module = torch.get_device_module(self.device)

        k_size, v_size = self.get_kv_size_bytes()
        
        
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )
        kv_logger.info(
            f"KV Cache is allocated. K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB, #layers: {layer_num}, head dim = {self.head_dim}, num head is {head_num}"
        )

        self.stop_event = threading.Event()
        self.alloc_job_queue = queue.Queue()
        self.alloc_thread = threading.Thread(target=self._alloc_worker_loop)
        self.alloc_thread.start()
        # self.alloc_lock = threading.Lock()

    def close(self):
        self.stop_event.set()
        self.alloc_job_queue.join()
        self.alloc_thread.join()
           
    def _alloc_worker_loop(self):
        while not self.stop_event.is_set():
            try:
                select_index = self.alloc_job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            try:
                if isinstance(select_index, torch.Tensor):
                    select_index_cpu = select_index.cpu()
                    select_index_cpu = select_index_cpu.tolist()
                else:
                    select_index_cpu =  select_index
        
                self.kv_pool.prepare_access(select_index_cpu, self.token_size, self.layer_num, self.distance_layer, 1)

            finally:
                self.alloc_job_queue.task_done()
                
                
    def _create_buffers(self):
        with self.memory_saver_adapter.region():
            
            runtime_id = os.getenv("RUNTIME_ID")
            if runtime_id is None:
                raise RuntimeError("runtime_id is not set")
            else:
                print("[App]: runtime_id:", runtime_id)
    
            slot_id = self.kv_pool.init_model_device(self.head_num * self.head_dim, self.layer_num * 2, self.store_dtype, int(runtime_id), self.tp_size)
            
            print(f"[App]: Register kv pool with slot_id {slot_id}")
            alloc_size = self.layer_num * (self.size + self.page_size) * self.head_num * self.head_dim
            distance_layer = (self.size + self.page_size) * self.head_num * self.head_dim
            token_size = self.head_num * self.head_dim
            is_k = 1
            self.k_buffer_pt =  self.kv_pool.allocate_kv(alloc_size, token_size, self.layer_num, distance_layer, is_k)
            is_k = 0
            self.v_buffer_pt =  self.kv_pool.allocate_kv(alloc_size, token_size, self.layer_num, distance_layer, is_k)

    def _clear_buffers(self):
        # del self.k_buffer
        # del self.v_buffer
        self.kv_pool.delete_all()

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer_pt")
        assert hasattr(self, "v_buffer_pt")
        
        size_in_bytes = torch.tensor(0, dtype=self.store_dtype).element_size()
        size_bytes = self.layer_num * (self.size + self.page_size) * self.head_num * self.head_dim * size_in_bytes
        return size_bytes, size_bytes

    # below: for disagg
   
    def get_flat_data(self, indices):
        raise NotImplementedError
        

    @debug_timing
    def transfer(self, indices, flat_data):
        raise NotImplementedError

    def transfer_per_layer(self, indices, flat_data, layer_id):
        raise NotImplementedError


    def get_key_buffer(self, layer_id: int):
        # if not self.k_buffer_cache:
        #     self.k_buffer_cache = {}
            
        if layer_id in self.k_buffer_cache:
            key_tensor = self.k_buffer_cache[layer_id]
        else: 
            layer_index = layer_id * (self.size + self.page_size) * self.head_num * self.head_dim
            shape = [(self.size + self.page_size), self.head_num, self.head_dim]

            key_tensor = self.kv_pool.get_buffer_tensor(
                layer_id, layer_index, self.store_dtype, self.dtype, shape, 1
            )
            # key_tensor = self.kv_pool.get_buffer_tensor(
            #     layer_index, self.store_dtype, self.dtype, shape, 1
            # )
            self.k_buffer_cache[layer_id] = key_tensor

        return key_tensor


    def get_value_buffer(self, layer_id: int):
        # if not self.v_buffer_cache:
        #     self.v_buffer_cache = {}
        
        if layer_id in self.v_buffer_cache:
            value_tensor = self.v_buffer_cache[layer_id]
            
        else: 
            layer_index = layer_id * (self.size + self.page_size) * self.head_num * self.head_dim
            shape = [(self.size + self.page_size), self.head_num, self.head_dim]
            value_tensor = self.kv_pool.get_buffer_tensor(layer_id, layer_index, self.store_dtype, self.dtype, shape, 0) #is_k = 0
            #value_tensor = self.kv_pool.get_buffer_tensor(layer_index, self.store_dtype, self.dtype, shape, 0)
            self.v_buffer_cache[layer_id] = value_tensor
        return value_tensor

    def get_kv_buffer(self, layer_id: int):
        
        start_time = time.perf_counter()
        ret = self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1e6
        # kv_logger.info(
        #     f"get() took {elapsed_us:.2f} µs"
        # )
        return ret

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        layer_id = layer.layer_id
    
        layer_index = layer_id * (self.size + self.page_size) * self.head_num * self.head_dim
        token_size = self.head_num * self.head_dim
        start_time = time.perf_counter()
        # print("===== write_kv DEBUG START =====")
        # print(f"layer_index: {layer_index}")
        # print(f"loc shape: {loc.shape}, indices.min: {loc.min()}, max: {loc.max()}")
        # # print(f"loc: {loc}")
        # print(f"cache_k shape: {cache_k.shape}, dtype: {cache_k.dtype}, device: {cache_k.device}")
       
        # get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
        # get_ptr.restype = ctypes.c_void_p
        # get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]

        # ptr_val = get_ptr(self.k_buffer_pt, None)

        # print(f"k_buffer_pt (hex): {hex(ptr_val)}")
        # print("===== write_kv DEBUG END =====")
        
        self.alloc_job_queue.join() # waits for all queued work so far

        self.kv_pool.write_kv(layer_index, loc, cache_k, token_size, self.store_dtype, 1) # k
        self.kv_pool.write_kv(layer_index, loc, cache_v, token_size, self.store_dtype, 0) # v
        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1e6
        # kv_logger.info(
        #     f"set() took {elapsed_us:.2f} µs"
        # )

# @torch.compile
def fused_downcast(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dtype: torch.dtype,
    store_dtype: torch.dtype,
    max_fp8: float,
    min_fp8: float,
):
    cache_k = cache_k / k_scale
    cache_k = torch.clamp(cache_k, min_fp8, max_fp8)
    cache_v = cache_v / v_scale
    cache_v = torch.clamp(cache_v, min_fp8, max_fp8)
    cache_k = cache_k.to(dtype)
    cache_v = cache_v.to(dtype)
    cache_k = cache_k.view(store_dtype)
    cache_v = cache_v.view(store_dtype)
    return cache_k, cache_v


# This compiled version is slower in the unit test
# python3 -m unittest test_bench_serving.TestBenchServing.test_offline_throughput_non_stream_small_batch_size
@torch.compile(dynamic=True, backend=get_compiler_backend())
def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype, store_dtype):
    dst_1[loc] = src_1.to(dtype).view(store_dtype)
    dst_2[loc] = src_2.to(dtype).view(store_dtype)

