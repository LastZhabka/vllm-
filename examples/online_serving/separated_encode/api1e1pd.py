# api_proxy.py
import asyncio
import os
import uuid
from typing import AsyncIterator, Optional, cast, Dict, Any, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json, time
import uvicorn
import argparse
import logging
from vllm.inputs import PromptType, SingletonPrompt, TextPrompt, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs  
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         parse_chat_messages,
                                         apply_hf_chat_template,
                                         resolve_chat_template_content_format)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

E_RANK = 0
PD_RANK = 1
encode_engine: Optional[AsyncLLM] = None
prefill_decode_engine: Optional[AsyncLLM] = None


def format_streaming_response(output) -> str:  
    """Convert vLLM output to OpenAI streaming format."""  
    # Implementation depends on your specific format requirements  
    # This is a simplified example  
    chunk = {  
        "id": output.request_id,  
        "object": "chat.completion.chunk",  
        "created": int(time.time()),  
        "choices": [{  
            "index": 0,  
            "delta": {"content": output.outputs[0].text if output.outputs else ""},  
            "finish_reason": "stop" if output.finished else None  
        }]  
    }  
    return f"data: {json.dumps(chunk)}\n\n"  


async def format_prompt(messages):
    list_of_messages = [
        cast(list[ChatCompletionMessageParam], messages)
    ]
    tokenizer = await encode_engine.get_tokenizer(None)
    model_config = await encode_engine.get_model_config()
    resolved_content_format = resolve_chat_template_content_format(
        None,
        None,
        "auto",
        tokenizer,
        model_config=model_config,
    )

    _chat_template_kwargs: dict[str, Any] = dict(
        chat_template=None,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
    )

    prompts: list[Union[TokensPrompt, TextPrompt]] = []

    for msgs in list_of_messages:
        # NOTE: _parse_chat_message_content_parts() currently doesn't
        # handle mm_processor_kwargs, since there is no implementation in
        # the chat message parsing for it.
        conversation, mm_data = parse_chat_messages(
            msgs,
            model_config,
            tokenizer,
            content_format=resolved_content_format,
        )
        prompt_str = apply_hf_chat_template(
            tokenizer=tokenizer,
            conversation=conversation,
            model_config=model_config,
            **_chat_template_kwargs,
        )

        prompt = TextPrompt(prompt=prompt_str)

        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        prompts.append(prompt)
    return prompts[0]

async def forward_streaming_request(
    request_data: dict,
    request_id: str
) -> AsyncIterator[str]:
    messages = request_data.get("messages", [])
    # convert from dict to vLLM
    prompt = await format_prompt(messages)
    sampling_params = ChatCompletionRequest(**request_data).to_sampling_params(
        max_tokens=128,
        logits_processor_pattern=encode_engine.model_config.logits_processor_pattern,
        default_sampling_params=encode_engine.model_config.get_diff_sampling_param(),
    )
    async for _ in encode_engine.generate(
        request_id=request_id,  
        prompt=prompt,  
        sampling_params=sampling_params  
    ):  
        pass
    try:
        async for output in prefill_decode_engine.generate(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params
        ):
            chunk = format_streaming_response(output)  
            yield chunk 
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    global encode_engine, prefill_decode_engine
    os.environ["VLLM_USE_V1"] = "1"  
    encode_engine_args = AsyncEngineArgs(
        model = "/workspace/helper/Qwen2.5-VL-3B-Instruct",
        gpu_memory_utilization = 0.2,
        max_num_seqs = 128,
        instance_type = "encode",
        connector_workers_num = 8,
        epd_rank = 0,
    )
    prefill_decode_engine_args = AsyncEngineArgs(
        model = "/workspace/helper/Qwen2.5-VL-3B-Instruct",
        gpu_memory_utilization = 0.6,
        max_num_seqs = 128,
        instance_type = "prefill+decode",
        connector_workers_num = 8,
        epd_rank = 1,
    )
    encode_engine = AsyncLLM.from_engine_args(
        encode_engine_args)
    prefill_decode_engine = AsyncLLM.from_engine_args(
        prefill_decode_engine_args)    

@app.on_event("shutdown")
async def shutdown_event():
    global encode_engine, prefill_decode_engine
    if encode_engine:  
        encode_engine.shutdown()  
    if prefill_decode_engine:  
        prefill_decode_engine.shutdown()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests."""
    try:
        # Parse request data
        request_data = await request.json()
        
        # Generate request ID if not provided
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add rank information to request ID
        request_id = f"{request_id}|{E_RANK}|{PD_RANK}"
        
        # Check if streaming is requested
        is_streaming = request_data.get("stream", False)
        
        if is_streaming:
            # Return streaming response
            return StreamingResponse(
                forward_streaming_request(request_data, request_id),
                media_type="text/event-stream"
            )
        else:
            logger.error(f"Only streaming requests")
            raise
                
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():  
    """List available models."""  
    try:  
        # Get model info from decode engine  
        model_config = prefill_decode_engine.model_config  
        return {  
            "object": "list",  
            "data": [{  
                "id": model_config.model,  
                "object": "model",  
                "created": int(time.time()),  
                "owned_by": "vllm"  
            }]  
        }  
    except Exception as e:  
        logger.error(f"Error fetching models: {e}")  
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    health_status = {
        "proxy": "healthy",
        "encode_server": "healthy",
        "prefill_decode_server": "healthy"
    }
    
    return health_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Proxy for distributed vLLM servers")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")
    parser.add_argument("--e-rank", type=int, default=0, help="Encode server rank")
    parser.add_argument("--pd-rank", type=int, default=1, help="Prefill/decode server rank")
    args = parser.parse_args()

    E_RANK = args.e_rank
    PD_RANK = args.pd_rank
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Encode: (rank {E_RANK})")
    logger.info(f"Prefill/Decode: (rank {PD_RANK})")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop"
    )