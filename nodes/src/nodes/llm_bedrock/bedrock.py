# =============================================================================
# MIT License
# Copyright (c) 2026 Aparavi Software AG
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

"""
Bedrock binding for the ChatLLM.
"""

from typing import Any, Dict, Optional
from ai.common.chat import ChatBase
from ai.common.config import Config
from ai.common.llm_native_stream import PROMPT_CACHE_PREFIX_VAR, apply_prompt_cache_breakpoint
from langchain_aws import ChatBedrock
from langchain_aws.utils import parse_model_provider

# Cross-region inference prefixes the node adds itself. An id that already starts with
# one of them, or an ARN, is sent as typed.
_REGION_PREFIXES = ('us.', 'eu.', 'apac.')

# ARN resource types whose resource id names the vendor (``foundation-model/anthropic.``,
# ``inference-profile/us.anthropic.``). A provisioned, custom or imported model ARN and
# an application inference profile carry an opaque id instead.
_ARN_TYPES_NAMING_THE_VENDOR = ('foundation-model', 'inference-profile')


def region_prefix(region: Optional[str]) -> str:
    """Return the cross-region inference prefix for an AWS *region*."""
    region = region or ''
    if region.startswith('eu'):
        return 'eu.'
    if region.startswith('ap'):
        return 'apac.'
    return 'us.'


def resolve_model_id(model: Optional[str], region: Optional[str]) -> str:
    """Return the model id to send to Bedrock for *model* in *region*.

    A plain id gets the prefix of the region. An ARN, an id that already carries a
    prefix, and an empty id are returned as typed. The save-time probe and the chat
    client both build their id here, so the two cannot drift apart again.
    """
    model = model or ''
    if not model or model.startswith('arn:') or model.startswith(_REGION_PREFIXES):
        return model
    return region_prefix(region) + model


def arn_provider(model_id: str) -> Optional[str]:
    """Return the vendor named by a foundation-model or inference-profile ARN, else None.

    langchain-aws picks the request format from the vendor and cannot read it from an
    ARN, so it refuses to build a client for one unless the caller passes ``provider``.
    """
    if not model_id.startswith('arn:'):
        return None
    # arn:partition:service:region:account:resource-type/resource-id; the id itself may
    # hold colons (``...-v1:0``), so split off the first five fields only.
    resource = model_id.split(':', 5)[-1]
    resource_type, _, resource_id = resource.partition('/')
    if resource_type not in _ARN_TYPES_NAMING_THE_VENDOR or not resource_id:
        return None
    return parse_model_provider(resource_id) or None


def bedrock_model_args(model: Optional[str], region: Optional[str]) -> Dict[str, Any]:
    """Return the ``model`` (and ``provider`` for an ARN that names one) for ChatBedrock."""
    model_id = resolve_model_id(model, region)
    args: Dict[str, Any] = {'model': model_id}
    provider = arn_provider(model_id)
    if provider:
        args['provider'] = provider
    return args


# Claude 3 models on Bedrock that accept a cache marker. Every Claude from 4 on does.
# The other Claude 3 models and Claude 2 reject the request with a ValidationException,
# and Claude 3.5 Sonnet v2 has caching only as a preview, so they never get the marker.
_CACHEABLE_CLAUDE_3 = ('claude-3-7-sonnet', 'claude-3-5-haiku')


def _accepts_cache_marker(model_id: str) -> bool:
    """Return True when Bedrock accepts a cache_control marker for *model_id*.

    Listing the old models that lack caching, rather than the new ones that have it,
    keeps every Claude released after this code cached without a change here.
    """
    model_id = (model_id or '').lower()
    if 'anthropic.claude-' not in model_id:
        return False
    if 'claude-3-' in model_id:
        return any(name in model_id for name in _CACHEABLE_CLAUDE_3)
    return 'claude-v' not in model_id and 'claude-instant' not in model_id


def _count_cache_as_input(usage: Any) -> None:
    """Add the cached tokens to ``input_tokens`` in a ChatBedrock ``usage_metadata``.

    Bedrock's input count leaves out tokens read from or written to the cache (AWS:
    total input = inputTokens + cacheReadInputTokens + cacheWriteInputTokens).
    LangChain counts the whole prompt as input_tokens, and the token meter subtracts
    the cache detail from it, so a prompt read from the cache was metered as almost
    no input. langchain-aws adds them for the Converse API (Nova) but not for
    InvokeModel, which ChatBedrock uses for every other model.
    """
    if not isinstance(usage, dict):
        return
    details = usage.get('input_token_details') or {}
    try:
        cached = int(details.get('cache_read') or 0) + int(details.get('cache_creation') or 0)
        if cached:
            usage['input_tokens'] = int(usage.get('input_tokens') or 0) + cached
            usage['total_tokens'] = usage['input_tokens'] + int(usage.get('output_tokens') or 0)
    except (TypeError, ValueError):
        pass  # metering is best effort; never fail a reply that was already paid for


class _ChatBedrock(ChatBedrock):
    """ChatBedrock that marks the cacheable start of a Claude prompt and meters cached tokens.

    Both request paths (invoke and stream) build the Anthropic message list and pass
    it to these two methods, so the marker is placed in one spot for both.
    """

    def _prepare_input_and_invoke(self, prompt=None, system=None, messages=None, **kwargs):
        self._mark_cache_prefix(messages)
        return super()._prepare_input_and_invoke(prompt=prompt, system=system, messages=messages, **kwargs)

    def _prepare_input_and_invoke_stream(self, prompt=None, system=None, messages=None, **kwargs):
        self._mark_cache_prefix(messages)
        return super()._prepare_input_and_invoke_stream(prompt=prompt, system=system, messages=messages, **kwargs)

    def _mark_cache_prefix(self, messages) -> None:
        if messages and _accepts_cache_marker(self.model_id):
            apply_prompt_cache_breakpoint({'messages': messages}, PROMPT_CACHE_PREFIX_VAR.get())

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if not self.beta_use_converse_api:  # the Converse path already counts them
            for generation in result.generations:
                _count_cache_as_input(getattr(generation.message, 'usage_metadata', None))
        return result

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            if not self.beta_use_converse_api:
                _count_cache_as_input(getattr(chunk.message, 'usage_metadata', None))
            yield chunk


class Chat(ChatBase):
    """
    Create a Bedrock chat bot.
    """

    _llm: ChatBedrock

    # Bedrock's own caching of Claude prompts is best effort. The driver marks the end
    # of the prompt's unchanging start so that part is eligible on every call; models
    # other than Claude never get the marker.
    SUPPORTS_PROMPT_CACHE_PREFIX = True

    def __init__(self, provider: str, connConfig: Dict[str, Any], bag: Dict[str, Any]):
        """
        Initialize the Bedrock chat bot.
        """
        # Init the base
        super().__init__(provider, connConfig, bag)

        # Get the nodes configuration
        config = Config.getNodeConfig(provider, connConfig)

        # Get the AWS access Key, don't save it
        accessKey = config.get('accessKey')

        # Get the AWS access Key, don't save it
        secretKey = config.get('secretKey')

        # Get the AWS region, save it
        self.region = config.get('region')

        # Get the llm
        self._llm = _ChatBedrock(
            **bedrock_model_args(self._model, self.region),
            aws_access_key_id=accessKey,
            aws_secret_access_key=secretKey,
            region=self.region,
            temperature=0,
            max_tokens=self._modelOutputTokens,
            custom_get_token_ids=self._estimate_token_ids,
        )

        # Save our chat class into the bag
        bag['chat'] = self

    @staticmethod
    def _estimate_token_ids(text: str) -> list:
        """Estimate token ids at about four characters a token.

        Without this, LangChain counts tokens with the GPT-2 tokenizer from the
        transformers package, which neither the engine nor this node installs (and
        which downloads the tokenizer on first use), so on an engine without it
        every chat failed before it was sent. The count feeds ChatBase's size
        warnings and, through getTokenCounter, nodes that cut documents to fit the
        model (summarization, preprocessor_llm). GPT-2 does not match these
        models' tokenizers either; the Anthropic node uses this same estimate.
        """
        return [0] * max(1, (len(text) + 3) // 4)
