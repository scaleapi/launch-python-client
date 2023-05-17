import importlib.util
import re
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, Mock

import pytest
from _pytest.assertion.rewrite import AssertionRewritingHook

from launch.model_bundle import ModelBundle
from launch.model_endpoint import AsyncEndpoint, ModelEndpoint

ROOT_DIR = Path(__file__).parent.parent

TEST_SKIP_MAGIC_STRING = "# test='skip'"


@pytest.fixture
def import_execute(request, tmp_work_path: Path):
    def _import_execute(module_name: str, source: str, rewrite_assertions: bool = False):
        if rewrite_assertions:
            loader = AssertionRewritingHook(config=request.config)
            loader.mark_rewrite(module_name)
        else:
            loader = None

        module_path = tmp_work_path / f"{module_name}.py"
        module_path.write_text(source)
        spec = importlib.util.spec_from_file_location("__main__", str(module_path), loader=loader)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

    return _import_execute


def extract_code_chunks(path: Path, text: str, offset: int):
    rel_path = path.relative_to(ROOT_DIR)
    for m_code in re.finditer(r"^```(.*?)$\n(.*?)^```", text, flags=re.M | re.S):
        prefix = m_code.group(1).lower()
        if not prefix.startswith(("py", "{.py")):
            continue

        start_line = offset + text[: m_code.start()].count("\n") + 1
        code = m_code.group(2)
        if TEST_SKIP_MAGIC_STRING in code:
            code = code.replace(TEST_SKIP_MAGIC_STRING, "")
            start_line += 1
            end_line = start_line + code.count("\n") + 1
            source = "__skip__"
        else:
            end_line = start_line + code.count("\n") + 1
            source = "\n" * start_line + code
        yield pytest.param(f"{path.stem}_{start_line}_{end_line}", source, id=f"{rel_path}:{start_line}-{end_line}")


def generate_code_chunks(*directories: str):
    for d in directories:
        for path in (ROOT_DIR / d).glob("**/*"):
            if path.suffix == ".py":
                code = path.read_text()
                for m_docstring in re.finditer(r'(^\s*)r?"""$(.*?)\1"""', code, flags=re.M | re.S):
                    start_line = code[: m_docstring.start()].count("\n")
                    docstring = dedent(m_docstring.group(2))
                    yield from extract_code_chunks(path, docstring, start_line)
            elif path.suffix == ".md":
                code = path.read_text()
                yield from extract_code_chunks(path, code, 0)


@pytest.fixture
def mock_dictionary():
    mock = MagicMock()
    mock.__getitem__.side_effect = lambda key: mock
    return mock


@pytest.fixture
def mock_async_endpoint() -> AsyncEndpoint:
    mock = Mock(spec=AsyncEndpoint)
    mock.model_endpoint = Mock(spec=ModelEndpoint)
    mock.model_endpoint.id = "test-endpoint"
    mock.status = Mock(return_value="READY")
    return mock


@pytest.fixture
def mock_model_bundle() -> ModelBundle:
    mock = Mock(spec=ModelBundle)
    mock.id = "test-bundle"
    return mock


@pytest.fixture
def mock_batch_job():
    return {"job_id": "test-batch-job", "status": "SUCCESS"}


@pytest.mark.parametrize("module_name,source_code", generate_code_chunks("launch", "docs"))
def test_docs_examples(
    module_name,
    source_code,
    import_execute,
    mocker,
    mock_dictionary,
    mock_model_bundle,
    mock_async_endpoint,
    mock_batch_job,
):
    mocker.patch("launch.connection.Connection", MagicMock())
    mocker.patch("launch.client.DefaultApi", MagicMock())
    mocker.patch("launch.model_endpoint.DefaultApi", MagicMock())
    mocker.patch("json.loads", MagicMock(return_value=mock_dictionary))
    mocker.patch("launch.model_bundle.ModelBundle.from_dict", MagicMock())
    mocker.patch("launch.model_endpoint.ModelEndpoint.from_dict", MagicMock())
    mocker.patch("launch.client.LaunchClient.get_model_bundle", MagicMock(return_value=mock_model_bundle))
    mocker.patch("launch.client.LaunchClient.get_model_endpoint", MagicMock(return_value=mock_async_endpoint))
    mocker.patch("launch.client.LaunchClient.create_model_bundle", MagicMock(return_value=mock_model_bundle))
    mocker.patch("launch.client.LaunchClient.create_model_endpoint", MagicMock(return_value=mock_async_endpoint))
    mocker.patch("launch.client.LaunchClient.get_batch_async_response", MagicMock(return_value=mock_batch_job))
    mocker.patch("launch.client.Connection.make_request", MagicMock(return_value=mock_dictionary))
    mocker.patch("launch.client.requests", MagicMock())
    mocker.patch("pydantic.BaseModel.parse_raw", MagicMock())

    if source_code == "__skip__":
        pytest.skip("test='skip' on code snippet")

    async def dont_aiosleep(t):
        pass

    async def dont_sleep(t):
        pass

    mocker.patch("asyncio.sleep", new=dont_aiosleep)
    mocker.patch("time.sleep", new=dont_sleep)

    try:
        import_execute(module_name, source_code, True)
    except Exception:
        raise
