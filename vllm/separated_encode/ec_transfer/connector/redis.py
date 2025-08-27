# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Literal, Optional

import msgpack_numpy
import numpy as np
import redis
from numpy.typing import NDArray

from vllm.config import VllmConfig
from vllm.separated_encode.ec_transfer.connector.template import (
    ECConnectorTemplate)
from vllm.logger import init_logger

logger = init_logger(__name__)

class RedisECConnector(ECConnectorTemplate):

    def __init__(self,
                 vllm_config: "VllmConfig",
                 intra_instance_type: Literal["scheduler", "model-runner"],
                 preallocate_callback: Optional[Callable[[str, int, int, str],
                                                         None]],
                 injection_callback: Optional[Callable[
                     [str, int, NDArray[np.float32], str], None]],
                 redis_host: str = "localhost",
                 redis_port: int = 6379):
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port)
        self.rank = vllm_config.epd_disagg_config.epd_rank
        super().__init__(
            vllm_config,
            intra_instance_type,
            preallocate_callback,
            injection_callback,
        )
    
    def _get_request_ranks(self, request_id: str):
        # request_id format: $ACTUAL_REQUEST_ID|$E_RANK|$PD_RANK
        result = request_id.split("|")
        return int(result[1]), int(result[2])

    def _send_prealloc_notification(self, request_id: str, input_id: int, 
                                    successful: bool, mm_hash: str) -> None:
        # PD -> E
        transfer_data = {
            "request_id": request_id, 
            "input_id": input_id, 
            "successful": successful,
            "mm_hash": mm_hash
        }
        rank = self._get_request_ranks(request_id)[0]
        logger.debug(f"Sent prealloc notification -> {rank}, {request_id}, {successful}")        
        self.redis_client.lpush(f"prealloc{rank}", 
                                msgpack_numpy.packb(transfer_data))

    def _send_encoder_cache_metas(
        self, request_id: str, input_id: int,
        num_encoder_tokens: int, mm_hash: str
    ) -> None:
        # E -> PD
        transfer_data = {
            "request_id": request_id,
            "input_id": input_id,
            "num_encoder_tokens": num_encoder_tokens,
            "mm_hash": mm_hash
        }
        rank = self._get_request_ranks(request_id)[1]
        logger.debug(f"Sent encode cache metadata -> {rank}, {request_id}")
        self.redis_client.lpush(f"cache_metas{rank}",
                                msgpack_numpy.packb(transfer_data))

    def _send_encoder_cache(
        self, request_id: str, input_id: int,
        encoder_cache: NDArray[np.float32], mm_hash: str) -> None:
        # E -> PD
        transfer_data = msgpack_numpy.packb({
            "request_id": request_id,
            "input_id": input_id,
            "encoder_cache": encoder_cache,
            "mm_hash": mm_hash
        })
        rank = self._get_request_ranks(request_id)[1]
        logger.debug(f"Sent encode cache -> {rank}, {request_id}")
        self.redis_client.lpush(f"cache{rank}", transfer_data)

    def _recv_prealloc_notification(
            self, maybe_send_cache_callback: Callable[[str, int, bool, str],
                                                      None]) -> None:
        transfered_data = self.redis_client.blpop(f"prealloc{self.rank}")[1]
        transfered_data = msgpack_numpy.unpackb(transfered_data, raw=False)
        request_id, input_id, successful, mm_hash = (
            transfered_data["request_id"],
            transfered_data["input_id"],
            transfered_data["successful"],
            transfered_data["mm_hash"]
        )
        logger.debug(f"Received prealloc notif -> {self.rank}, {request_id}")
        maybe_send_cache_callback(request_id, input_id, successful, mm_hash)

    def _recv_encoder_cache_metas(
            self, preallocate_callback: Callable[[str, int, int, str],
                                                 None]) -> None:
        transfered_data = self.redis_client.blpop(f"cache_metas{self.rank}")[1]
        transfered_data = msgpack_numpy.unpackb(transfered_data, raw=False)
        request_id, input_id, num_encoder_tokens, mm_hash = (
            transfered_data["request_id"], 
            transfered_data["input_id"],
            transfered_data["num_encoder_tokens"],
            transfered_data["mm_hash"]
        )
        logger.debug(f"Received encoder metadata -> {self.rank}, {request_id}")
        preallocate_callback(request_id, input_id, num_encoder_tokens, mm_hash)

    def _recv_encoder_cache(
        self, 
        injection_callback: Callable[[str, int, NDArray[np.float32], str],None]
    ) -> None:
        transfered_data = self.redis_client.blpop(f"cache{self.rank}")[1]
        transfered_data = msgpack_numpy.unpackb(transfered_data, raw=False)
        request_id, input_id, encoder_cache, mm_hash = (
            transfered_data["request_id"], 
            transfered_data["input_id"],
            transfered_data["encoder_cache"],
            transfered_data["mm_hash"]
        )
        logger.debug(f"Received encoder cache -> {self.rank}, {request_id}")
        injection_callback(request_id, input_id, encoder_cache, mm_hash)
