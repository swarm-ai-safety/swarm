"""Tests for AgentRxiv integration.

Tests cover:
- AgentRxivClient: API interactions
- AgentRxivServer: Server lifecycle management
- PDF export utilities
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from swarm.research.agentrxiv_server import (
    AgentRxivServer,
    AgentRxivServerError,
    ensure_agent_laboratory,
)
from swarm.research.pdf_export import (
    PDFExportError,
    check_pdflatex,
    extract_text_from_pdf,
    markdown_to_pdf,
    paper_to_pdf,
)
from swarm.research.platforms import (
    AgentRxivClient,
    Paper,
    SearchResult,
    get_client,
)


class TestAgentRxivClientInit:
    """Test AgentRxivClient initialization."""

    def test_default_base_url(self):
        """Default base URL is localhost:5000."""
        client = AgentRxivClient()
        assert client.base_url == "http://127.0.0.1:5000"

    def test_custom_base_url(self):
        """Can specify custom base URL."""
        client = AgentRxivClient(base_url="http://custom:8080")
        assert client.base_url == "http://custom:8080"

    def test_env_var_base_url(self):
        """Environment variable overrides default URL."""
        with patch.dict("os.environ", {"AGENTRXIV_URL": "http://env:9000"}):
            client = AgentRxivClient()
            assert client.base_url == "http://env:9000"

    def test_explicit_url_overrides_env(self):
        """Explicit URL parameter overrides environment variable."""
        with patch.dict("os.environ", {"AGENTRXIV_URL": "http://env:9000"}):
            client = AgentRxivClient(base_url="http://explicit:7000")
            assert client.base_url == "http://explicit:7000"

    def test_api_key_not_required(self):
        """AgentRxiv doesn't require API key."""
        client = AgentRxivClient()
        # Should not raise, API key is optional
        assert client.api_key is None


class TestAgentRxivClientHealthCheck:
    """Test health check functionality."""

    def test_health_check_success(self):
        """Health check returns True when server responds."""
        client = AgentRxivClient()
        with patch.object(client._session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            assert client.health_check() is True
            mock_get.assert_called_once_with("http://127.0.0.1:5000/", timeout=5)

    def test_health_check_failure_connection(self):
        """Health check returns False on connection error."""
        client = AgentRxivClient()
        with patch.object(client._session, "get") as mock_get:
            mock_get.side_effect = requests.ConnectionError()
            assert client.health_check() is False

    def test_health_check_failure_status(self):
        """Health check returns False on non-200 status."""
        client = AgentRxivClient()
        with patch.object(client._session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            assert client.health_check() is False


class TestAgentRxivClientSearch:
    """Test search functionality."""

    def test_search_parses_results(self):
        """Search returns properly parsed papers."""
        client = AgentRxivClient()
        mock_data = [
            {"id": 1, "filename": "test_paper.pdf", "text": "Abstract content here"},
            {"id": 2, "filename": "another_paper.pdf", "text": "More content"},
        ]

        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_data
            mock_request.return_value = mock_response

            result = client.search("test query")

            assert isinstance(result, SearchResult)
            assert len(result.papers) == 2
            assert result.query == "test query"
            assert result.papers[0].paper_id == "agentrxiv:1"
            assert result.papers[0].title == "test paper"
            assert "Abstract content" in result.papers[0].abstract

    def test_search_respects_limit(self):
        """Search limits results correctly."""
        client = AgentRxivClient()
        mock_data = [{"id": i, "filename": f"paper_{i}.pdf", "text": "x"} for i in range(10)]

        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_request.return_value = mock_response

            result = client.search("query", limit=3)
            assert len(result.papers) == 3

    def test_search_handles_error(self):
        """Search returns empty result on error."""
        client = AgentRxivClient()
        with patch.object(client, "_request") as mock_request:
            mock_request.side_effect = requests.ConnectionError()
            result = client.search("query")

            assert result.papers == []
            assert result.total_count == 0


class TestAgentRxivClientSubmit:
    """Test paper submission."""

    def test_submit_requires_pdf_path(self):
        """Submit fails without PDF path."""
        client = AgentRxivClient()
        paper = Paper(title="Test Paper", abstract="Abstract")

        result = client.submit(paper)

        assert result.success is False
        assert "pdf_path" in result.message.lower()

    def test_submit_uploads_pdf(self):
        """Submit uploads PDF file correctly."""
        client = AgentRxivClient()
        paper = Paper(title="My Research Paper", abstract="Abstract")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 mock content")
            pdf_path = f.name

        try:
            with patch.object(client._session, "post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                result = client.submit(paper, pdf_path=pdf_path)

                assert result.success is True
                assert "My Research Paper" in result.paper_id
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert "files" in call_args.kwargs
        finally:
            Path(pdf_path).unlink()

    def test_submit_handles_error(self):
        """Submit returns failure on error."""
        client = AgentRxivClient()
        paper = Paper(title="Test", abstract="Abstract")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            pdf_path = f.name

        try:
            with patch.object(client._session, "post") as mock_post:
                mock_post.side_effect = requests.ConnectionError("Server down")

                result = client.submit(paper, pdf_path=pdf_path)

                assert result.success is False
                assert "Server down" in result.message
        finally:
            Path(pdf_path).unlink()


class TestAgentRxivClientGetPaper:
    """Test paper retrieval."""

    def test_get_paper_with_prefix(self):
        """Get paper handles agentrxiv: prefix."""
        client = AgentRxivClient()
        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            result = client.get_paper("agentrxiv:123")

            assert result is not None
            mock_request.assert_called_with("GET", "http://127.0.0.1:5000/view/123")

    def test_get_paper_not_found(self):
        """Get paper returns None for 404."""
        client = AgentRxivClient()
        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_request.return_value = mock_response

            result = client.get_paper("999")
            assert result is None


class TestAgentRxivClientTriggerUpdate:
    """Test update trigger."""

    def test_trigger_update_success(self):
        """Trigger update returns True on success."""
        client = AgentRxivClient()
        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            assert client.trigger_update() is True

    def test_trigger_update_failure(self):
        """Trigger update returns False on failure."""
        client = AgentRxivClient()
        with patch.object(client, "_request") as mock_request:
            mock_request.side_effect = requests.ConnectionError()
            assert client.trigger_update() is False


class TestGetClientFactory:
    """Test get_client factory function."""

    def test_get_agentrxiv_client(self):
        """Factory returns AgentRxivClient for 'agentrxiv'."""
        client = get_client("agentrxiv")
        assert isinstance(client, AgentRxivClient)

    def test_unknown_platform_raises(self):
        """Factory raises for unknown platform."""
        with pytest.raises(ValueError, match="Unknown platform"):
            get_client("unknownplatform")


class TestAgentRxivServerInit:
    """Test AgentRxivServer initialization."""

    def test_default_settings(self):
        """Default server settings are sensible."""
        server = AgentRxivServer()
        assert server.port == 5000
        assert server.uploads_dir == Path("./agentrxiv_papers")
        assert server.is_running is False

    def test_custom_port(self):
        """Can specify custom port."""
        server = AgentRxivServer(port=8080)
        assert server.port == 8080
        assert server.base_url == "http://127.0.0.1:8080"

    def test_custom_uploads_dir(self):
        """Can specify custom uploads directory."""
        server = AgentRxivServer(uploads_dir="/tmp/papers")
        assert server.uploads_dir == Path("/tmp/papers")


class TestAgentRxivServerLifecycle:
    """Test server lifecycle management."""

    def test_is_running_no_process(self):
        """is_running returns False when no process."""
        server = AgentRxivServer()
        assert server.is_running is False

    def test_start_without_agent_lab_raises(self):
        """Start raises when Agent Laboratory not installed."""
        server = AgentRxivServer(agent_lab_path="/nonexistent/path")
        with pytest.raises(AgentRxivServerError, match="not installed"):
            server.start()

    def test_context_manager_stops_on_exit(self):
        """Context manager stops server on exit."""
        server = AgentRxivServer()
        server._process = MagicMock()
        server._process.poll.return_value = None  # Running

        with patch.object(server, "start"):
            with patch.object(server, "stop") as mock_stop:
                with server:
                    pass  # Context manager enter/exit
                mock_stop.assert_called_once()


class TestAgentRxivServerSeedPapers:
    """Test paper seeding functionality."""

    def test_seed_nonexistent_dir(self):
        """Seeding nonexistent dir returns 0."""
        server = AgentRxivServer()
        count = server.seed_papers("/nonexistent/path")
        assert count == 0

    def test_seed_copies_pdfs(self):
        """Seeding copies PDF files."""
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as uploads_dir:
                # Create mock PDFs
                pdf1 = Path(source_dir) / "paper1.pdf"
                pdf2 = Path(source_dir) / "paper2.pdf"
                pdf1.write_bytes(b"%PDF-1.4 content1")
                pdf2.write_bytes(b"%PDF-1.4 content2")

                server = AgentRxivServer(uploads_dir=uploads_dir)
                server.uploads_dir.mkdir(parents=True, exist_ok=True)

                count = server.seed_papers(source_dir)

                assert count == 2
                assert (Path(uploads_dir) / "paper1.pdf").exists()
                assert (Path(uploads_dir) / "paper2.pdf").exists()


class TestEnsureAgentLaboratory:
    """Test Agent Laboratory installation helper."""

    def test_already_installed(self):
        """Returns path if already installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app_py = Path(tmpdir) / "app.py"
            app_py.write_text("# Flask app")

            result = ensure_agent_laboratory(tmpdir)
            assert result == Path(tmpdir)

    def test_clone_on_missing(self):
        """Clones repository if not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            install_path = Path(tmpdir) / "AgentLab"

            with patch("subprocess.run") as mock_run:
                # Simulate git clone failure
                mock_run.return_value = MagicMock(returncode=1, stderr="clone failed")

                with pytest.raises(AgentRxivServerError, match="Failed to clone"):
                    ensure_agent_laboratory(install_path)

                # Verify git clone was attempted
                mock_run.assert_called_once()
                call_args = mock_run.call_args
                assert "git" in call_args[0][0]
                assert "clone" in call_args[0][0]


class TestCheckPdflatex:
    """Test pdflatex availability check."""

    def test_pdflatex_available(self):
        """Returns True when pdflatex works."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_pdflatex() is True

    def test_pdflatex_not_found(self):
        """Returns False when pdflatex not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert check_pdflatex() is False

    def test_pdflatex_timeout(self):
        """Returns False on timeout."""
        with patch("subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired("pdflatex", 5)
            assert check_pdflatex() is False


class TestPaperToPdf:
    """Test paper_to_pdf function."""

    def test_no_source_raises(self):
        """Raises ValueError when paper has no source."""
        paper = Paper(title="Test", abstract="Abstract")
        with pytest.raises(ValueError, match="no LaTeX source"):
            paper_to_pdf(paper)

    def test_no_pdflatex_raises(self):
        """Raises PDFExportError when pdflatex unavailable."""
        paper = Paper(title="Test", source="\\documentclass{article}")
        with patch("swarm.research.pdf_export.check_pdflatex") as mock_check:
            mock_check.return_value = False
            with pytest.raises(PDFExportError, match="pdflatex not found"):
                paper_to_pdf(paper)


class TestMarkdownToPdf:
    """Test markdown_to_pdf function."""

    def test_no_pandoc_raises(self):
        """Raises PDFExportError when pandoc unavailable."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with pytest.raises(PDFExportError, match="pandoc not found"):
                markdown_to_pdf("# Test", title="Test")


class TestExtractTextFromPdf:
    """Test extract_text_from_pdf function."""

    def test_missing_file_raises(self):
        """Raises PDFExportError for missing file or missing library."""
        # The function checks for pypdf before file existence,
        # so error could be either "PDF not found" or "pypdf required"
        with pytest.raises(PDFExportError):
            extract_text_from_pdf("/nonexistent/file.pdf")

    def test_missing_library_raises(self):
        """Raises PDFExportError when pypdf not available."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            pdf_path = f.name

        try:
            with patch.dict("sys.modules", {"pypdf": None}):
                with patch("builtins.__import__") as mock_import:
                    def import_side_effect(name, *args, **kwargs):
                        if name in ("pypdf", "PyPDF2"):
                            raise ImportError()
                        return MagicMock()
                    mock_import.side_effect = import_side_effect

                    with pytest.raises(PDFExportError, match="pypdf"):
                        extract_text_from_pdf(pdf_path)
        finally:
            Path(pdf_path).unlink()


class TestAgentRxivIntegration:
    """Integration tests for AgentRxiv workflow."""

    def test_search_and_paper_id_format(self):
        """Paper IDs from search have correct format."""
        client = AgentRxivClient()
        mock_data = [{"id": 42, "filename": "test.pdf", "text": "content"}]

        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_request.return_value = mock_response

            result = client.search("query")

            assert result.papers[0].paper_id.startswith("agentrxiv:")
            # ID can be used with get_paper
            paper_id = result.papers[0].paper_id
            assert "42" in paper_id

    def test_submit_workflow(self):
        """Full submit workflow with PDF."""
        client = AgentRxivClient()
        paper = Paper(
            title="Test Paper",
            abstract="This is a test abstract",
            categories=["agentrxiv"],
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test content")
            pdf_path = f.name

        try:
            with patch.object(client._session, "post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                result = client.submit(paper, pdf_path=pdf_path)

                assert result.success is True
                assert result.paper_id
        finally:
            Path(pdf_path).unlink()
