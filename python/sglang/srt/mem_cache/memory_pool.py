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
import csv
import threading
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
import traceback

logger = logging.getLogger(__name__)


kv_logger = logging.getLogger("kv_logger")
kv_logger.setLevel(logging.INFO)

kv_logger.propagate = False

background_sync_csv_lock = threading.Lock()
mem_free_csv_lock = threading.Lock()
write_csv_lock = threading.Lock()
read_csv_lock = threading.Lock()
prepare_access_csv_lock = threading.Lock()

def read_config_from_file():
    config = {}
    """Read configuration values from config.txt file."""
    config_path = f"/home/sean/diss/virtualize_llm/config.txt"

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            print(f"[Config] Successfully read configuration from {config_path}")
        else:
            print(f"[Config] Config file not found at {config_path}, using defaults")
            
    except Exception as e:
        print(f"[Config] Error reading config file: {e}, using defaults")
    
    return config

# Read configuration from file
config = read_config_from_file()
MEMORY_LOCATION = config["MEMORY_LOCATION"]
# INPUT_LEN = int(config["INPUT_LEN"])
# OUTPUT_LEN = int(config["OUTPUT_LEN"])
BATCH_SIZE = config["BATCH_SIZE"]
DURATION = config["DURATION"]
RPS = config["RPS"]
METHOD = config["METHOD"]

print(f"[Config] Using configuration: MEMORY_LOCATION={MEMORY_LOCATION}, BATCH_SIZE={BATCH_SIZE}, DURATION={DURATION}, RPS={RPS}")


# Initialising csv files
def initialize_csv_files():
    """Initialize CSV files with headers if they don't exist."""
    # Use absolute path to avoid permission/path issues
    base_path = f"/home/sean/diss/virtualize_llm/experiment_results/{METHOD}/" + f"{BATCH_SIZE}_batch_size/data"
    
    csv_files = {
        f"{base_path}/prepare_access_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv": ["operation", "latency_us", "num_tokens", "num_layers", "Request ID"],
        f"{base_path}/background_synchronisation_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv": ["operation", "latency_us", "num_tokens", "Layer ID", "Request ID"],
        # f"{base_path}/mem_free_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv": ["operation", "latency_us", "num_tokens", "num_layers"],
        f"{base_path}/write_kv_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv": ["operation", "num_tokens", "layer_id", "latency_us"],
        f"{base_path}/read_kv_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv": ["operation", "layer_id", "latency_us"]
    }
    
    # Create the directory structure if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    for csv_file, headers in csv_files.items():
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"[CSV Init] Created {csv_file} with headers: {headers}")

initialize_csv_files()

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
        self.current_request_id = 0
        self.accumulated_latency = 0.0
        self.first_request = True
        self.prev_num_tokens = None
        self.batch_prepare_access_time = 0
        self.batch_layers_prepared = 0
      
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
    # Allocates space for tokens in the virtual address space
    # need_size is the number of new tokens that need to be allocated space in the virtual KV cache
    # def alloc(self, need_size: int):
    #     # free_slots is a list of token indices in the virtual address space that are free for allocation
    #     if need_size > len(self.free_slots) or self.not_available:
    #         # 
    #         return None
        
    #     start_time = time.perf_counter()
    #     # Select the first need_size indices from free_slots (i.e. from the free_slots in virtual memory)
    #     select_index = self.free_slots[:need_size]
    #     self.free_slots = self.free_slots[need_size:]
        
    #     self._kvcache.alloc_job_queue.put(select_index)
        
    #     end_time = time.perf_counter()
    #     elapsed_us = (end_time - start_time) * 1e6
    #     # kv_logger.info(
    #     #     f"alloc1 took {elapsed_us:.2f} µs"
    #     # )
    #     self.log_allocated_size()

    #     return select_index
    
    def alloc(self, need_size: int):
        if need_size > len(self.free_slots) or self.not_available:
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        if isinstance(select_index, torch.Tensor):
            select_index_cpu = select_index.cpu().tolist()
        else:
            select_index_cpu = select_index
        num_tokens = len(select_index_cpu)

        start_time_prep = time.perf_counter()
        self._kvcache.kv_pool.prepare_access(
            select_index_cpu, self.token_size, self._kvcache.layer_num, self.distance_layer, 0
        )
        end_time_prep = time.perf_counter()
        elapsed_us_prep = (end_time_prep - start_time_prep) * 1e6

        with prepare_access_csv_lock:
            with open(
                f"/home/sean/diss/virtualize_llm/experiment_results/{METHOD}/{BATCH_SIZE}_batch_size/data/prepare_access_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv",
                'a', newline=''
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    "prepare_access_all_layers",
                    elapsed_us_prep,
                    num_tokens,
                    self._kvcache.layer_num
                ])

        return select_index


    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []  
         
            
    def free(self, free_index):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.free_slots = torch.cat((self.free_slots, free_index))
                
            if isinstance(free_index, torch.Tensor):
                indices = free_index.cpu().tolist()
            else:
                indices = list(free_index)
            
            #start_time_free = time.perf_counter()
            self._kvcache.kv_pool.mem_free(indices, self.token_size, self._kvcache.layer_num, self.distance_layer)
            #end_time_free = time.perf_counter()
            #elapsed_us = (end_time_free - start_time_free) * 1e6
            # with mem_free_csv_lock:
            #     with open(f"/home/sean/diss/virtualize_llm/experiment_results/{METHOD}/mem_free_{MEMORY_LOCATION}_input_{INPUT_LEN}_output_{OUTPUT_LEN}.csv", 'a', newline='') as csv_file:
            #         writer = csv.writer(csv_file)
            #         writer.writerow(["mem_free", elapsed_us, len(indices), self._kvcache.layer_num])
            
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
        # tensor parallelism size (number of GPUs that the model is split across)
        tp_size: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        
        # Add batch write tracking variables
        self.batch_write_start_time = None
        self.batch_write_total_time = 0
        self.batch_layers_written = 0
        self.batch_tokens_count = 0
        self.batch_write_lock = threading.Lock()
        self.current_request_id = 0
        
        # Add batch read tracking variables
        self.batch_read_start_time = None
        self.batch_read_total_time = 0
        self.batch_layers_read = 0
        self.batch_read_tokens_count = 0
        self.batch_read_lock = threading.Lock()
        
        print(f"[DEBUG] MHATokenToKVPool has been initialised with size: {size}, page_size: {page_size}, dtype: {dtype}, head_num: {head_num}, head_dim: {head_dim}, layer_num: {layer_num}, device: {device}, tp_size: {tp_size}")
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
        # Create a job queue for the background thread to monitor
        # These jobs are token indices that are about to be written to or read from.
        # So the background thread will back these token indices with physical memory in prepare access.
        #self.alloc_job_queue = queue.Queue()
        #self.alloc_thread = threading.Thread(target=self._alloc_worker_loop)
        #self.alloc_thread.start()
        #self.alloc_lock = threading.Lock()


    # def close(self):
    #     self.stop_event.set()
    #     self.alloc_job_queue.join()
    #     self.alloc_thread.join()
           
    # def _alloc_worker_loop(self):
    #     # Accumulate latency per request
    #     accumulated_latency = 0.0
    #     current_request_id = 0
    #     first_request = True
    #     # Token sizes that indicate a new request
    #     prefill_sizes = {64, 128, 256, 512}

    #     while not self.stop_event.is_set():
    #         try:
    #             select_index = self.alloc_job_queue.get(timeout=0.1)
    #         except queue.Empty:
    #             continue

    #         try:
    #             if isinstance(select_index, torch.Tensor):
    #                 select_index_cpu = select_index.cpu().tolist()
    #             else:
    #                 select_index_cpu = select_index
    #             num_tokens = len(select_index_cpu)

    #             start_time_prep = time.perf_counter()
    #             self.kv_pool.prepare_access(select_index_cpu, self.token_size, self.layer_num, self.distance_layer, 1)
    #             end_time_prep = time.perf_counter()
    #             elapsed_us_prep = (end_time_prep - start_time_prep) * 1e6

    #             # If this is a prefill (start of a new request)
    #             if num_tokens in prefill_sizes:
    #                 if not first_request:
    #                     # Write the accumulated latency for the previous request
    #                     with prepare_access_csv_lock:
    #                         with open(f"/home/sean/diss/virtualize_llm/experiment_results/{METHOD}/{BATCH_SIZE}_batch_size/data/prepare_access_{MEMORY_LOCATION}_input_{INPUT_LEN}_output_{OUTPUT_LEN}.csv", 'a', newline='') as csv_file:
    #                             writer = csv.writer(csv_file)
    #                             writer.writerow(["prepare_access", accumulated_latency, prev_num_tokens, current_request_id])
    #                 else:
    #                     first_request = False

    #                 # Start new request
    #                 current_request_id += 1
    #                 accumulated_latency = elapsed_us_prep
    #                 prev_num_tokens = num_tokens
    #             else:
    #                 # Accumulate latency for current request
    #                 accumulated_latency += elapsed_us_prep

    #         finally:
    #             self.alloc_job_queue.task_done()

    #     # On shutdown, flush the last request's accumulated latency
    #     if not first_request:
    #         with prepare_access_csv_lock:
    #             with open(f"/home/sean/diss/virtualize_llm/experiment_results/{METHOD}/{BATCH_SIZE}_batch_size/data/prepare_access_{MEMORY_LOCATION}_input_{INPUT_LEN}_output_{OUTPUT_LEN}.csv", 'a', newline='') as csv_file:
    #                 writer = csv.writer(csv_file)
    #                 writer.writerow(["prepare_access", accumulated_latency, prev_num_tokens, current_request_id])
                
                
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
        with self.batch_read_lock:
            # Initialize batch tracking on first layer
            if layer_id == 0:
                self.batch_read_start_time = time.perf_counter()
                self.batch_read_total_time = 0
                self.batch_layers_read = 0

        start_time = time.perf_counter()
        ret = self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1e6
        
        with self.batch_read_lock:
            self.batch_read_total_time += elapsed_us
            self.batch_layers_read += 1
            
            # If this is the last layer, record the total read time
            if self.batch_layers_read == self.layer_num:
                with read_csv_lock:
                    with open(f"/home/sean/diss/virtualize_llm/experiment_results/{METHOD}/{BATCH_SIZE}_batch_size/data/read_kv_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv", 'a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        # Write in this format: Operation,Layer ID, Latency across all layers.
                        writer.writerow(["read_kv_all_layers", layer_id, self.batch_read_total_time])
                        
                
                # Reset for next batch
                self.batch_read_start_time = None
                self.batch_read_total_time = 0
                self.batch_layers_read = 0
                
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
        
        # Blocks main thread until the alloc_job_queue is empty
        # Waits for all prepare_access calls in the background thread queue to finish before proceeding
        # set_kv_buffer happens during token generation, so this latency directly affects the time users wait for tokens
        join_start = time.perf_counter()
        #self.alloc_job_queue.join() # waits for all queued work so far
        join_end = time.perf_counter()
        join_elapsed_us = (join_end - join_start) * 1e6
        
        with self.batch_write_lock:
            # Initialize batch tracking on first layer
            if layer_id == 0:
                self.batch_write_start_time = time.perf_counter()
                self.batch_write_total_time = 0
                self.batch_layers_written = 0
                self.batch_tokens_count = len(loc)
                # Initialize background sync tracking
                self.batch_sync_total_time = 0
                
                if self.batch_tokens_count == 128 or self.batch_tokens_count == 64 or self.batch_tokens_count == 256 or self.batch_tokens_count == 512:
                    self.current_request_id += 1

            # Accumulate background sync time
            self.batch_sync_total_time += join_elapsed_us

        start_time_write = time.perf_counter()
        self.kv_pool.write_kv(layer_index, loc, cache_k, token_size, self.store_dtype, 1, layer_id, self.layer_num) # k
        self.kv_pool.write_kv(layer_index, loc, cache_v, token_size, self.store_dtype, 0, layer_id, self.layer_num) # v
        end_time_write = time.perf_counter()
        elapsed_us = (end_time_write - start_time_write) * 1e6
        
        with self.batch_write_lock:
            self.batch_write_total_time += elapsed_us
            self.batch_layers_written += 1

            # If this is the last layer, record both write time AND background sync time
            if self.batch_layers_written == self.layer_num:
                # Log total write time across all layers
                with write_csv_lock:
                    with open(f"/home/sean/diss/virtualize_llm/experiment_results/{METHOD}/{BATCH_SIZE}_batch_size/data/write_kv_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv", 'a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([
                            "write_kv_all_layers",
                            self.batch_tokens_count,
                            layer_id,  # This is the last layer's ID, you may want to use a special value or the request ID
                            self.batch_write_total_time
                        ])
                # ... (background sync logging and reset)

                # Log accumulated background synchronization time across all layers
                with background_sync_csv_lock:
                    with open(f"/home/sean/diss/virtualize_llm/experiment_results/{METHOD}/{BATCH_SIZE}_batch_size/data/background_synchronisation_{MEMORY_LOCATION}_duration_{DURATION}_rps_{RPS}.csv", 'a', newline='') as csv_file:
                                writer = csv.writer(csv_file)
                                writer.writerow(["background_synchronisation_all_layers", self.batch_sync_total_time, self.batch_tokens_count, layer_id])
                        
                # Reset for next batch
                self.batch_write_start_time = None
                self.batch_write_total_time = 0
                self.batch_layers_written = 0
                self.batch_tokens_count = 0
                self.batch_sync_total_time = 0

@torch.compile 
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


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):      
        
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        with memory_saver_adapter.region():
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.kv_buffer = [
                torch.zeros(
                    (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                    dtype=self.store_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]

        self.layer_transfer_counter = None
        self.page_size = page_size

        kv_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, KV size: {kv_size / GB:.2f} GB"
        )

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += np.prod(kv_cache.shape) * kv_cache.dtype.itemsize
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)

        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope
        )

    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        return torch.stack([self.kv_buffer[i][indices] for i in range(self.layer_num)])

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        for i in range(self.layer_num):
            self.kv_buffer[i][indices] = flat_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        self.kv_buffer[layer_id - self.start_layer][indices] = flat_data


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        with memory_saver_adapter.region():
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

            # [size, head_num, heavy_channel_num] for each layer
            self.label_buffer = [
                torch.zeros(
                    (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label

    def get_flat_data(self, indices):
        pass

    def transfer(self, indices, flat_data):
        pass

    def transfer_per_layer(self, indices, flat_data, layer_id):
        pass


class MemoryStateInt(IntEnum):
    IDLE = 0
    RESERVED = 1
    PROTECTED = 2
    SYNCED = 3
    BACKUP = 4


def synchronized(debug_only=False):
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if (not debug_only) or self.debug:
                return func(self, *args, **kwargs)
                with self.lock:
                    return func(self, *args, **kwargs)
            else:
                return True

        return wrapper

    return _decorator


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        pin_memory: bool,
        device: str,
        page_size: int,
    ):
        self.device_pool = device_pool
        self.dtype = device_pool.store_dtype
        self.pin_memory = pin_memory
        self.device = device
        self.page_size = page_size
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align the host memory pool size to the page size
        self.size = self.size - (self.size % self.page_size)

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        if requested_bytes > host_mem.available - ten_gb:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{host_mem.available / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.debug = logger.isEnabledFor(logging.DEBUG)
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data(self, indices):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_flat_data_by_layer(self, indices, layer_id):
        raise NotImplementedError()

    @abc.abstractmethod
    def assign_flat_data(self, indices, flat_data):
        raise NotImplementedError()

    @synchronized()
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized()
    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        if self.debug:
            self.mem_state[select_index] = MemoryStateInt.RESERVED

        return select_index

    @synchronized()
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        if self.debug:
            self.mem_state[indices] = MemoryStateInt.IDLE
        return len(indices)

    @synchronized(debug_only=True)
    def get_state(self, indices: torch.Tensor) -> MemoryStateInt:
        assert len(indices) > 0, "The indices should not be empty"
        states = self.mem_state[indices]
        assert (
            states == states[0]
        ).all(), "The memory slots should have the same state {}".format(states)
        return MemoryStateInt(states[0].item())

    @synchronized(debug_only=True)
    def is_reserved(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.RESERVED

    @synchronized(debug_only=True)
    def is_protected(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def is_synced(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.SYNCED

    @synchronized(debug_only=True)
    def is_backup(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.BACKUP

    @synchronized(debug_only=True)
    def update_backup(self, indices: torch.Tensor):
        if not self.is_synced(indices):
            raise ValueError(
                f"The host memory slots should be in SYNCED state before turning into BACKUP. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.BACKUP

    @synchronized(debug_only=True)
    def update_synced(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.SYNCED

    @synchronized(debug_only=True)
    def protect_write(self, indices: torch.Tensor):
        if not self.is_reserved(indices):
            raise ValueError(
                f"The host memory slots should be RESERVED before write operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def protect_load(self, indices: torch.Tensor):
        if not self.is_backup(indices):
            raise ValueError(
                f"The host memory slots should be in BACKUP state before load operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized(debug_only=True)
    def complete_io(self, indices: torch.Tensor):
        if not self.is_protected(indices):
            raise ValueError(
                f"The host memory slots should be PROTECTED during I/O operations. "
                f"Current state: {self.get_state(indices)}"
            )
        self.mem_state[indices] = MemoryStateInt.SYNCED


class MHATokenToKVPoolHost(HostKVCache):
    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            device_pool, host_to_device_ratio, host_size, pin_memory, device, page_size
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def init_kv_buffer(self):
        return torch.empty(
            (2, self.layer_num, self.size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @debug_timing
    def transfer(self, indices, flat_data):
        # backup prepared data from device to host
        self.kv_buffer[:, :, indices] = flat_data.to(
            device=self.device, non_blocking=False
        )

    def get_flat_data(self, indices):
        return self.kv_buffer[:, :, indices]

    def get_flat_data_by_layer(self, indices, layer_id):
        return self.kv_buffer[:, layer_id - self.start_layer, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, :, indices] = flat_data

    def write_page_all_layers(self, host_indices, device_indices, device_pool):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = host_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            for j in range(self.layer_num):
                self.kv_buffer[0, j, h_index : h_index + self.page_size].copy_(
                    device_pool.k_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )
                self.kv_buffer[1, j, h_index : h_index + self.page_size].copy_(
                    device_pool.v_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )

    def load_page_per_layer(self, host_indices, device_indices, device_pool, layer_id):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = host_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            device_pool.k_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    0, layer_id - self.start_layer, h_index : h_index + self.page_size
                ],
                non_blocking=True,
            )
            device_pool.v_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    1, layer_id - self.start_layer, h_index : h_index + self.page_size
                ],
                non_blocking=True,
            )


class MLATokenToKVPoolHost(HostKVCache):
    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            device_pool, host_to_device_ratio, host_size, pin_memory, device, page_size
        )

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num

        return (
            (self.kv_lora_rank + self.qk_rope_head_dim)
            * 1
            * self.dtype.itemsize
            * self.layer_num
        )

    def init_kv_buffer(self):
        return torch.empty(
            (
                self.layer_num,
                self.size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @debug_timing
    def transfer(self, indices, flat_data):
        # backup prepared data from device to host
        self.kv_buffer[:, indices] = flat_data.to(
            device=self.device, non_blocking=False
        )

    def get_flat_data(self, indices):
        return self.kv_buffer[:, indices]

    def get_flat_data_by_layer(self, indices, layer_id):
        return self.kv_buffer[layer_id - self.start_layer, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, indices] = flat_data

    def write_page_all_layers(self, host_indices, device_indices, device_pool):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = host_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            for j in range(self.layer_num):
                self.kv_buffer[j, h_index : h_index + self.page_size].copy_(
                    device_pool.kv_buffer[j][d_index : d_index + self.page_size],
                    non_blocking=True,
                )

    def load_page_per_layer(self, host_indices, device_indices, device_pool, layer_id):
        device_indices_cpu = device_indices[:: self.page_size].cpu()
        for i in range(len(device_indices_cpu)):
            h_index = host_indices[i * self.page_size]
            d_index = device_indices_cpu[i]
            device_pool.kv_buffer[layer_id - self.start_layer][
                d_index : d_index + self.page_size
            ].copy_(
                self.kv_buffer[
                    layer_id - self.start_layer, h_index : h_index + self.page_size
                ],
                non_blocking=True,
            )