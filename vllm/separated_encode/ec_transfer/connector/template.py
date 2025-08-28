# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal, Optional

import torch

from vllm.config import EPDDisaggConfig, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

class ECConnectorTemplate(ABC):
    """
    Abstraction for the communication between the E instance and
    P(or PD) instance, all encoder cache communication is handled
    by this class.

    ECConnector communication handling is executed in separate thread,
    the cache injection, preallocation and allocation logic is handled
    by the gpu model runner/scheduler function. 
    
    Send and receive logic are handled by specific implementations like
    RedisECConnector, note that all _recv tasks are created in advance
    on the class startup, and the number of _recv tasks is maintained
    so it'll be better to remove timeout from you _recv functions 
    implementation. 
    
    Also the ECConnector move the encoder_cache dict in itself to handle
    send encoder cache task.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: Optional[torch.device],
        intra_instance_type: Literal["scheduler", "model-runner"],
        preallocate_callback: Optional[Callable[[str, int, int, str], None]],
        injection_callback: Optional[Callable[[str, int, torch.Tensor, str],
                                              None]],
    ):
        callback_mapping = {
            ("encode", "scheduler"): (None, None),
            ("encode", "model-runner"):
            (self._recv_prealloc_notification, self._maybe_send_encoder_cache),
            ("prefill", "scheduler"):
            (self._recv_encoder_cache_metas, preallocate_callback),
            ("prefill", "model-runner"):
            (self._recv_encoder_cache, injection_callback),
            ("prefill+decode", "scheduler"):
            (self._recv_encoder_cache_metas, preallocate_callback),
            ("prefill+decode", "model-runner"): (self._recv_encoder_cache,
                                                 injection_callback)
        }
        self.device = device
        self.dtype = vllm_config.model_config.dtype

        self.epd_disagg_config: EPDDisaggConfig
        self.intra_instance_type: Literal["scheduler", "model-runner"]
        self.inter_instance_type: Literal["encode", "prefill",
                                          "prefill+decode"]
        self.encoder_cache: dict[str, dict[int, torch.Tensor]]
        self.send_executors: ThreadPoolExecutor
        self.recv_executors: ThreadPoolExecutor

        # Instance type and configs:
        self.epd_disagg_config = vllm_config.epd_disagg_config
        self.inter_instance_type = self.epd_disagg_config.instance_type
        self.intra_instance_type = intra_instance_type

        # Initialize main transfer processing components:
        self.send_tasks_queue: queue.Queue = queue.Queue()
        self.send_executors = ThreadPoolExecutor(
            max_workers=self.epd_disagg_config.connector_workers_num
        )

        # Sanity check
        assert self.epd_disagg_config.connector_workers_num > 0

        # Arif: max_workers num must match with limiting semaphore value
        # otherwise receive busy loop will infinitely create tasks for
        # the self.recv_executors  
        self.recv_executors = ThreadPoolExecutor(
            max_workers=self.epd_disagg_config.connector_workers_num + 1
        )
        self.send_worker = threading.Thread(target=self._send_event_loop)
        self.recv_worker = threading.Thread(target=self._recv_event_loop)
        self.target_recv_callback = callback_mapping.get(
            (self.inter_instance_type, self.intra_instance_type))
        
        
        self.limiting_semaphore = threading.Semaphore(
            self.epd_disagg_config.connector_workers_num + 1
        )

        # Used on model runner of encode instance:
        if (self.intra_instance_type == "model-runner"
                and self.inter_instance_type == "encode"):
            self.use_cache_lock: threading.Lock = threading.Lock()
            self.cache_to_send: set = set()
            self.cache_to_skip: set = set()
            self.encoder_cache = {}
            self.transfered_ids_lock: threading.Lock = threading.Lock()
            self.transfered_ids = []

        self.send_worker.start()
        self.recv_worker.start()

    @abstractmethod
    def _send_prealloc_notification(self, request_id: str, input_id: int, 
                                    successful: bool, mm_hash: str) -> None:
        """Send a pre-allocation completion notification.

        This method sends a notification to signal that the pre-allocation of
        space for an encoder cache, identified by request_id and input_id,
        has been completed on the P(or PD) instance.

        Args:
            request_id: id of the encoder cache's request.
            input_id: index of the mm input amoung request's mm inputs
            successful: indicates whether we need to send the encoder cache or not
            mm_hash: hash of the mm input
            
        """
        pass

    @abstractmethod
    def _send_encoder_cache_metas(self, request_id: str, input_id: int,
                                  num_encoder_tokens: int, mm_hash: str) -> None:
        """Send the metadata of an encoder cache.

        This method is used to transfer the encoder cache's metadata.

        Args:
            request_id: id of the encoder cache's request.
            input_id: index of the mm input amoung request's mm inputs
            num_encoder_tokens: size of the encoder cache
            mm_hash: hash of the mm input
        """
        pass

    @abstractmethod
    def _send_encoder_cache(
        self, request_id: str, input_id: int,
        encoder_cache: torch.Tensor, mm_hash: str
    ) -> None:
        """Send the encoder cache.

        This method sends the computed encoder cache in NumPy float type
        array. 

        Args:
            request_id: id of the encoder cache's request.
            input_id: index of the mm input amoung request's mm inputs
            encoder_cache: cache produced by vision model, in np array form
            mm_hash: hash of the mm input
        """
        pass

    @abstractmethod
    def _recv_prealloc_notification(
            self, maybe_send_cache_callback: Callable[[str, int, bool, str],
                                                      None]) -> None:
        """Receive a pre-allocation completion notification.

        This method invoke maybe_send_cache callback for any received
        pre-allocation notification. Note that you don't need to call 
        it immediately, you can delay the invocation of the callback,
        also this function is called in advance on the init startup.
        
        Check the receiving logic of RedisECConnector and recv event loop
        for more details.

        Args:
            maybe_send_cache_callback: A callback function within the ec 
                connector. This function either schedules encoder cache
                sending or adds the requested encoder cache to the set of 
                pending/ignored requests.   
        """
        pass

    @abstractmethod
    def _recv_encoder_cache_metas(
            self, preallocate_callback: Callable[[str, int, int, str],
                                                 None]) -> None:
        """Receives the encoder cache and calls preallocate callback

        This method invokes the preallocate callback for any received 
        encoder cache. Note that you don't need to call 
        it immediately, you can delay the invocation of the callback,
        also this function is called in advance on the init startup.
        
        Check the receiving logic of RedisECConnector and recv event loop
        for more details.

        Args:
            preallocate_callback: A callback function within the scheduler. 
                This function preallocates space for encoder cache in the
                encoder cache manager within the scheduler.  
        """
        pass

    @abstractmethod
    def _recv_encoder_cache(
        self, 
        injection_callback: Callable[[str, int, torch.Tensor, str],None]
    ) -> None:
        """Receives the encoder cache and calls injection callback

        This method invokes the injection callback for any received 
        encoder cache. Note that you don't need to call 
        it immediately, you can delay the invocation of the callback,
        also this function is called in advance on the init startup.
        
        Check the receiving logic of RedisECConnector and recv event loop
        for more details.

        Args:
            injection_callback: A callback function within the model runner. 
                This function injects encoder cache into the encoder_cache
                dictionary within the model runner.
        """
        pass

    def add_encoder_cache(self, request_id: str, input_id: int,
                          encoder_cache: torch.Tensor, mm_hash: str):
        """Add an encoder cache to the EC connector.

        This method adds the encoder cache to the self.encoder_cache dictionary
        if the encoder cache is not already present in the set of pending
        requested encoder caches.
        
        Args:
            request_id: id of the encoder cache's request.
            input_id: index of the mm input amoung request's mm inputs
            encoder_cache: encoder cache in numpy array form
        """
        with self.use_cache_lock:
            if (request_id, input_id) in self.cache_to_send:
                self.schedule_send_encoder_cache(request_id=request_id,
                                                input_id=input_id,
                                                encoder_cache=encoder_cache,
                                                mm_hash=mm_hash)
                self.cache_to_send.remove((request_id, input_id))
            elif (request_id, input_id) in self.cache_to_skip:
                with self.transfered_ids_lock:
                    self.transfered_ids.append((request_id, input_id))
                self.cache_to_skip.remove((request_id, input_id))
            else:
                if request_id not in self.encoder_cache:
                    self.encoder_cache[request_id] = {}
                self.encoder_cache[request_id][input_id] = encoder_cache

    def _maybe_send_encoder_cache(
        self, request_id: str, input_id: int, successful: bool, mm_hash: str
    ):
        """Sends the encoder cache or adds it to the pending send set

        This method schedules the task of sending the encoder cache if it was 
        already been calculated. If the cache is not available, the method adds 
        the request to the set of pending sends.

        Args:
            request_id: id of the encoder cache's request.
            input_id: index of the mm input amoung request's mm inputs
            successful: indicates whether we need to send the encoder cache or not
            mm_hash: hash of the mm input
        """
        with self.use_cache_lock:
            if (request_id in self.encoder_cache
                    and input_id in self.encoder_cache[request_id]):
                if successful:
                    self.schedule_send_encoder_cache(
                        request_id, input_id,
                        self.encoder_cache.get(request_id).get(input_id),
                        mm_hash)
                else:
                    with self.transfered_ids_lock:
                        self.transfered_ids.append((request_id, input_id))
                self.encoder_cache[request_id].pop(input_id)
                if not self.encoder_cache[request_id]:
                    self.encoder_cache.pop(request_id)
            else:
                if successful:
                    self.cache_to_send.add((request_id, input_id))
                else:
                    self.cache_to_skip.add((request_id, input_id))

    def _send_event_loop(self, ):
        """Run receive event loop 
        
        This method runs event loop for send tasks.
        """
        try:
            while True:
                callback, args = self.send_tasks_queue.get()
                self.send_executors.submit(callback, *args)
        except Exception as e:
            raise ConnectionError("Error during send event loop.") from e

    def _limiting_wrapper(self, callback: Callable, arg: Callable):
        """Wrapper function to limit the number of workers """
        with self.limiting_semaphore:
            callback(arg)

    def _recv_event_loop(self, ):
        """Run receive event loop 
        
        This method runs event loop for receive tasks and ensures that 
        the number of requested parallel receives is limited by
        $max_connector_workers.
        """
        try:
            if self.target_recv_callback[0] is None:
                return
            while True:
                callback, arg = self.target_recv_callback
                with self.limiting_semaphore:
                    self.recv_executors.submit(self._limiting_wrapper,
                                               callback, arg)
        except Exception as e:
            raise ConnectionError("Error during recv event loop") from e

    def schedule_send_prealloc_notification(self, request_id: str, input_id: int, 
                                            successful: bool, mm_hash: str) -> None:
        """Schedule preallocate completion notification sending

        This method schedules the task of sending preallocate completion
        notification to the encoder model runner(instance E).

        Args:
            request_id: id of the encoder cache's request.
            input_id: index of the mm input amoung request's mm inputs
            successful: indicates whether we need to send the encoder cache or not
            mm_hash: hash of the mm input
        """
        self.send_tasks_queue.put_nowait(
            (self._send_prealloc_notification, 
                (request_id, input_id, successful, mm_hash)))

    def schedule_send_encoder_cache_metadata(self, request_id: str,
                                             input_id: int,
                                             num_encoder_tokens: int,
                                             mm_hash: str) -> None:
        """Schedule encoder cache metadata sending

        This method schedules the task of sending encoder cache's metadata
        for the encoder cache space preallocation.

        Args:
            request_id: id of the encoder cache's request.
            input_id: index of the mm input amoung request's mm inputs
            num_encoder_tokens: size of the encoder cache
            mm_hash: hash of the mm input
        """
        self.send_tasks_queue.put_nowait(
            (self._send_encoder_cache_metas, (request_id, input_id,
                                              num_encoder_tokens, mm_hash)))

    def schedule_send_encoder_cache(
        self, request_id: str, input_id: int,
        encoder_cache: torch.Tensor, mm_hash: str
    ) -> None:
        """Schedule encoder cache sending

        This method schedules the task of sending encoder cache.

        Args:
            request_id: id of the encoder cache's request.
            input_id: index of the mm input amoung request's mm inputs
            encoder_cache: cache produced by vision model, in np array form
        """
        self.send_tasks_queue.put_nowait(
            (self._finish_wrapper, (self._send_encoder_cache, request_id,
                                    input_id, encoder_cache, mm_hash)))

    def _finish_wrapper(
        self, callback: Callable, request_id: str, input_id: int,
        encoder_cache: torch.Tensor, mm_hash: str
    ):

        callback(request_id, input_id, encoder_cache, mm_hash)
        with self.transfered_ids_lock:
            self.transfered_ids.append((request_id, input_id))

    def get_transfered_ids(self, ):
        with self.transfered_ids_lock:
            transfered_ids = self.transfered_ids
            self.transfered_ids = []
            return transfered_ids

    def finish_request(self, req_id):
        pass