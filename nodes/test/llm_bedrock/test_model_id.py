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

"""The model id the Bedrock node sends: region prefix, prefixed ids and ARNs (#2531).

The save-time probe (IGlobal.validateConfig) and the chat client (bedrock.Chat) used
to build the id separately. The probe left ARNs and already-prefixed ids alone, the
client always added the region prefix, so a config that passed the probe went out as
``us.us.anthropic...`` or ``us.arn:aws:bedrock:...`` at run time. Both now build the
id through one function, and an ARN that names its vendor also carries ``provider``,
which langchain-aws needs before it accepts an ARN at all. No key or network is used.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

pytest.importorskip('langchain_aws')

import langchain_aws  # noqa: E402

from ai.common.config import Config  # noqa: E402

_NODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'nodes', 'llm_bedrock')
_MOD_PATH = os.path.join(_NODE_DIR, 'bedrock.py')

_SONNET = 'anthropic.claude-sonnet-4-5-20250929-v1:0'
_PREFIXED = 'us.' + _SONNET
_FOUNDATION_ARN = 'arn:aws:bedrock:us-east-1::foundation-model/' + _SONNET
_PROFILE_ARN = 'arn:aws:bedrock:us-east-1:123456789012:inference-profile/' + _PREFIXED
_PROVISIONED_ARN = 'arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123xyz'


def _load_driver():
    spec = importlib.util.spec_from_file_location('_llm_bedrock_model_id_test', _MOD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(model: str, region: str = 'us-east-1') -> dict:
    return {
        'model': model,
        'accessKey': 'AKIDEXAMPLE',
        'secretKey': 'secret-example',
        'region': region,
        'modelTotalTokens': 200000,
        'modelOutputTokens': 8192,
    }


def _chat(monkeypatch, model: str, region: str = 'us-east-1'):
    config = _config(model, region)
    monkeypatch.setattr(Config, 'getNodeConfig', staticmethod(lambda *a, **k: dict(config)))
    return _load_driver().Chat('bedrock', {}, {})


@pytest.mark.parametrize(
    'region, expected',
    [
        ('us-east-1', 'us.' + _SONNET),
        ('eu-central-1', 'eu.' + _SONNET),
        ('ap-southeast-2', 'apac.' + _SONNET),
        ('ca-central-1', 'us.' + _SONNET),
    ],
)
def test_a_plain_id_gets_the_prefix_of_the_region(region, expected):
    assert _load_driver().resolve_model_id(_SONNET, region) == expected


@pytest.mark.parametrize('model', [_PREFIXED, 'eu.' + _SONNET, 'apac.' + _SONNET, _FOUNDATION_ARN, _PROVISIONED_ARN])
def test_a_prefixed_id_or_an_arn_is_sent_as_typed(model):
    assert _load_driver().resolve_model_id(model, 'us-east-1') == model


def test_an_empty_id_stays_empty():
    assert _load_driver().resolve_model_id('', 'us-east-1') == ''
    assert _load_driver().resolve_model_id(None, 'us-east-1') == ''


@pytest.mark.parametrize(
    'model, provider',
    [
        (_FOUNDATION_ARN, 'anthropic'),
        (_PROFILE_ARN, 'anthropic'),
        ('arn:aws:bedrock:us-east-1::foundation-model/meta.llama3-3-70b-instruct-v1:0', 'meta'),
        (_PROVISIONED_ARN, None),
        ('arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123xyz', None),
        (_SONNET, None),
        (_PREFIXED, None),
    ],
)
def test_the_vendor_is_read_from_an_arn_that_names_it(model, provider):
    driver = _load_driver()
    assert driver.arn_provider(model) == provider
    args = driver.bedrock_model_args(model, 'us-east-1')
    assert args.get('provider') == provider
    assert args['model'] == driver.resolve_model_id(model, 'us-east-1')


def test_the_chat_client_sends_a_prefixed_id_as_typed(monkeypatch):
    """Before: ``us.us.anthropic...``, an id Bedrock does not know."""
    chat = _chat(monkeypatch, _PREFIXED)

    assert chat._llm.model_id == _PREFIXED
    assert chat._llm._get_provider() == 'anthropic'


def test_the_chat_client_sends_an_arn_as_typed_with_its_vendor(monkeypatch):
    """Before: ``us.arn:aws:bedrock:...`` with ``arn`` as the vendor."""
    chat = _chat(monkeypatch, _FOUNDATION_ARN)

    assert chat._llm.model_id == _FOUNDATION_ARN
    assert chat._llm._get_provider() == 'anthropic'


def test_the_chat_client_still_adds_the_prefix_to_a_plain_id(monkeypatch):
    chat = _chat(monkeypatch, _SONNET, region='eu-west-1')

    assert chat._llm.model_id == 'eu.' + _SONNET
    assert chat._llm._get_provider() == 'anthropic'


def _load_node_package():
    """Load the node as a package so IGlobal's relative import of bedrock.py resolves."""
    spec = importlib.util.spec_from_file_location(
        '_llm_bedrock_node_under_test', os.path.join(_NODE_DIR, '__init__.py'), submodule_search_locations=[_NODE_DIR]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    try:
        spec.loader.exec_module(package)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return package


@pytest.fixture
def probe(monkeypatch):
    """Run IGlobal.validateConfig against a ChatBedrock stand-in; yield what it was built with."""
    built: list = []
    warnings: list = []

    class _Recorder:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def invoke(self, prompt):
            return prompt

    package = _load_node_package()
    monkeypatch.setattr(sys.modules[package.IGlobal.__module__], 'warning', warnings.append)
    monkeypatch.setattr(sys.modules['depends'], 'depends', lambda *a, **k: None)

    def run(model: str, region: str = 'us-east-1'):
        config = _config(model, region)
        monkeypatch.setattr(Config, 'getNodeConfig', staticmethod(lambda *a, **k: dict(config)))
        instance = package.IGlobal()
        instance.glb = types.SimpleNamespace(logicalType='llm_bedrock', connConfig={})
        # The stand-in is in place only while the probe runs, so a chat client built
        # afterwards in the same test still uses the real class.
        with monkeypatch.context() as patched:
            patched.setattr(langchain_aws, 'ChatBedrock', _Recorder)
            instance.validateConfig()
        return built[-1], list(warnings)

    yield run
    sys.modules.pop('_llm_bedrock_node_under_test', None)


def test_the_probe_and_the_chat_client_send_the_same_id(monkeypatch, probe):
    """The two paths build the id with one function, so what the probe checks is what runs."""
    for model, region in (
        (_SONNET, 'us-east-1'),
        (_SONNET, 'eu-west-1'),
        (_PREFIXED, 'us-east-1'),
        (_FOUNDATION_ARN, 'us-east-1'),
    ):
        kwargs, warnings = probe(model, region)
        chat = _chat(monkeypatch, model, region)

        assert warnings == []
        assert kwargs['model'] == chat._llm.model_id
        assert kwargs.get('provider') == (chat._llm.provider or None)


def test_the_probe_passes_the_vendor_for_an_arn(probe):
    kwargs, warnings = probe(_FOUNDATION_ARN)

    assert warnings == []
    assert kwargs['model'] == _FOUNDATION_ARN
    assert kwargs['provider'] == 'anthropic'
