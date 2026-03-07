from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.services import paddle_prewarm
from app.convert.ocr.routing import ROUTE_KIND_REMOTE_DOC_PARSER


class _FakePrewarmClient:
    def __init__(self) -> None:
        self.called = False

    def _get_paddle_doc_parser(self) -> None:
        self.called = True


def test_paddle_prewarm_uses_doc_parser_route(monkeypatch) -> None:
    client = _FakePrewarmClient()
    captured: dict[str, str | int | None] = {}

    monkeypatch.setattr(
        paddle_prewarm,
        "resolve_paddle_doc_prewarm_config",
        lambda service_role=None: paddle_prewarm.PaddleDocPrewarmConfig(
            provider="siliconflow",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            model="PaddlePaddle/PaddleOCR-VL-1.5",
            max_side_px=2200,
            required=False,
            service_role="worker",
        ),
    )

    def _fake_create_remote_ocr_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr(
        paddle_prewarm,
        "create_remote_ocr_client",
        _fake_create_remote_ocr_client,
    )

    assert paddle_prewarm.run_paddle_doc_prewarm() is True
    assert client.called is True
    assert captured == {
        "requested_provider": "aiocr",
        "route_kind": ROUTE_KIND_REMOTE_DOC_PARSER,
        "ai_api_key": "test-key",
        "ai_base_url": "https://api.siliconflow.cn/v1",
        "ai_model": "PaddlePaddle/PaddleOCR-VL-1.5",
        "ai_provider": "siliconflow",
        "paddle_doc_max_side_px": 2200,
    }


def test_local_layout_prewarm_downloads_pp_doclayout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        paddle_prewarm,
        "resolve_local_paddle_layout_prewarm_config",
        lambda service_role=None: paddle_prewarm.LocalPaddleLayoutPrewarmConfig(
            model_name="PP-DocLayoutV3",
            service_role="worker",
        ),
    )

    class _FakePaddleX:
        @staticmethod
        def create_model(model_name: str) -> str:
            captured["model_name"] = model_name
            return "ok"

    monkeypatch.setitem(sys.modules, "paddlex", _FakePaddleX())

    assert paddle_prewarm.run_local_paddle_layout_prewarm() is True
    assert captured == {"model_name": "PP-DocLayoutV3"}
