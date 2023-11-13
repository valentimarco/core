import os
from typing import Optional, List, Any, Mapping, Dict
import requests
from langchain.llms.base import LLM
from langchain.llms.openai import OpenAI
from langchain.llms.ollama import Ollama
import httpx
from langchain.llms import Ollama
import langchain.llms.ollama as _ollama 
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from typing import Any, Iterator, List, Optional, AsyncIterator
from langchain.schema.output import GenerationChunk
from cat.log import log
from langchain.schema import LLMResult
class LLMDefault(LLM):
    @property
    def _llm_type(self):
        return ""

    def _call(self, prompt, stop=None):
        return "AI: You did not configure a Language Model. " \
               "Do it in the settings!"
    
    async def _acall(self, prompt, stop=None):
        return "AI: You did not configure a Language Model. " \
               "Do it in the settings!"


# elaborated from
# https://python.langchain.com/en/latest/modules/models/llms/examples/custom_llm.html
class LLMCustom(LLM):
    # endpoint where custom LLM service accepts requests
    url: str

    # optional key for authentication
    auth_key: str = ""

    # optional dictionary containing custom configuration
    options: Dict = {}

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            # run_manager: Optional[CallbackManagerForLLMRun] = None,
            run_manager: Optional[Any] = None,
    ) -> str:

        request_body = {
            "text": prompt,
            "auth_key": self.auth_key,
            "options": self.options
        }

        try:
            response_json = requests.post(self.url, json=request_body).json()
        except Exception as exc:
            raise ValueError("Custom LLM endpoint error "
                             "during http POST request") from exc

        generated_text = response_json["text"]

        return generated_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Identifying parameters."""
        return {
            "url": self.url,
            "auth_key": self.auth_key,
            "options": self.options
        }


class CustomOpenAI(OpenAI):
    url: str
    
    def __init__(self, **kwargs):
        model_kwargs = {
            'repeat_penalty': kwargs.pop('repeat_penalty'),
            'top_k': kwargs.pop('top_k')
        }
        
        stop = kwargs.pop('stop', None)
        if stop:
            model_kwargs['stop'] = stop.split(',')

        super().__init__(
                    openai_api_key=" ",  
                    model_kwargs=model_kwargs,
                    **kwargs
                )
        
        self.url = kwargs['url']
        self.openai_api_base = os.path.join(self.url, "v1")




class CustomOllama(Ollama):
    streaming: bool = True
    batch_size: int = 20
    
    def get_sub_prompts(
        self,
        params: Dict[str, Any],
        prompts: List[str],
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Get the sub prompts for llm call."""
        params = params["options"]
        
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params["num_ctx"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["num_ctx"] = self.max_tokens_for_prompt(prompts[0])
        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        return sub_prompts

    
    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        params = self._default_params
        params = {**params, **kwargs}
        print(params)
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        if not self.streaming:
            response = super().generate(prompts, stop, **kwargs)
            return response
             
        for _prompts in sub_prompts:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")

                generation: Optional[GenerationChunk] = None
                async for chunk in self._astream(
                    _prompts[0], stop, run_manager, **kwargs
                ):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
                choices.append(
                    [generation]
                )   
    
        return LLMResult(generations=choices)
                    
                    
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _ollama._stream_response_to_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                   await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )