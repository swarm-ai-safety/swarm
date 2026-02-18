"""Tests for Memori middleware integration."""

from unittest.mock import MagicMock, patch

from swarm.agents.memori_middleware import MemoriConfig, MemoriMiddleware


class TestMemoriConfig:
    """Tests for MemoriConfig dataclass."""

    def test_defaults(self):
        cfg = MemoriConfig()
        assert cfg.enabled is False
        assert cfg.db_path == ":memory:"
        assert cfg.entity_id is None
        assert cfg.process_id is None
        assert cfg.auto_wait is True
        assert cfg.decay_on_epoch is True

    def test_from_dict_minimal(self):
        cfg = MemoriConfig.from_dict({"enabled": True})
        assert cfg.enabled is True
        assert cfg.db_path == ":memory:"

    def test_from_dict_full(self):
        cfg = MemoriConfig.from_dict(
            {
                "enabled": True,
                "db_path": "/tmp/test.db",
                "entity_id": "ent_1",
                "process_id": "proc_1",
                "auto_wait": False,
                "decay_on_epoch": False,
            }
        )
        assert cfg.db_path == "/tmp/test.db"
        assert cfg.entity_id == "ent_1"
        assert cfg.process_id == "proc_1"
        assert cfg.auto_wait is False
        assert cfg.decay_on_epoch is False

    def test_from_dict_empty(self):
        cfg = MemoriConfig.from_dict({})
        assert cfg.enabled is False


class TestMemoriMiddleware:
    """Tests for MemoriMiddleware with mocked Memori."""

    def _make_middleware(self, **config_overrides):
        cfg = MemoriConfig(enabled=True, **config_overrides)
        return MemoriMiddleware(cfg, agent_id="test_agent")

    @patch("swarm.agents.memori_middleware.MemoriMiddleware._get_memori")
    def test_wrap_client_calls_register(self, mock_get_memori):
        mock_memori = MagicMock()
        mock_get_memori.return_value = mock_memori

        mw = self._make_middleware()
        fake_client = MagicMock()
        result = mw.wrap_client(fake_client)

        mock_memori.llm.register.assert_called_once_with(fake_client)
        assert result is fake_client

    @patch("swarm.agents.memori_middleware.MemoriMiddleware._get_memori")
    def test_wrap_client_unsupported_provider_logs_warning(self, mock_get_memori):
        mock_memori = MagicMock()
        mock_memori.llm.register.side_effect = TypeError("unsupported")
        mock_get_memori.return_value = mock_memori

        mw = self._make_middleware()
        fake_client = MagicMock()
        # Should not raise
        result = mw.wrap_client(fake_client)
        assert result is fake_client

    def test_on_epoch_boundary_river_no_session(self):
        """River agents (persistence=1.0) should NOT call new_session()."""
        mw = self._make_middleware()
        mw._memori = MagicMock()

        mw.on_epoch_boundary(epoch=2, epistemic_persistence=1.0)
        mw._memori.new_session.assert_not_called()

    def test_on_epoch_boundary_rain_calls_new_session(self):
        """Rain agents (persistence=0.0) should call new_session()."""
        mw = self._make_middleware()
        mw._memori = MagicMock()

        mw.on_epoch_boundary(epoch=2, epistemic_persistence=0.0)
        mw._memori.new_session.assert_called_once()

    def test_on_epoch_boundary_partial_calls_new_session(self):
        """Partial persistence (0.5) should call new_session()."""
        mw = self._make_middleware()
        mw._memori = MagicMock()

        mw.on_epoch_boundary(epoch=3, epistemic_persistence=0.5)
        mw._memori.new_session.assert_called_once()

    def test_on_epoch_boundary_decay_disabled(self):
        """decay_on_epoch=False should skip session rotation."""
        mw = self._make_middleware(decay_on_epoch=False)
        mw._memori = MagicMock()

        mw.on_epoch_boundary(epoch=1, epistemic_persistence=0.0)
        mw._memori.new_session.assert_not_called()

    def test_lazy_init_no_memori_before_use(self):
        """Memori instance should be None until first use."""
        mw = self._make_middleware()
        assert mw._memori is None

    def test_wait_for_augmentation_noop_when_uninitialized(self):
        """wait_for_augmentation should be safe when _memori is None."""
        mw = self._make_middleware()
        mw.wait_for_augmentation()  # Should not raise

    def test_close_clears_memori(self):
        mw = self._make_middleware(auto_wait=False)
        mw._memori = MagicMock()
        mw.close()
        assert mw._memori is None

    def test_close_waits_when_auto_wait(self):
        mw = self._make_middleware(auto_wait=True)
        mock_memori = MagicMock()
        mw._memori = mock_memori
        mw.close()
        mock_memori.augmentation.wait.assert_called_once()

    @patch.dict("sys.modules", {"memori": MagicMock()})
    def test_get_memori_lazy_init(self):
        """_get_memori should create a Memori instance on first call."""
        import sys

        mock_module = sys.modules["memori"]
        mock_memori_cls = MagicMock()
        mock_module.Memori = mock_memori_cls

        mw = self._make_middleware()
        result = mw._get_memori()

        mock_memori_cls.assert_called_once_with(
            entity_id="test_agent",
            process_id="swarm-test_agent",
            db_path=":memory:",
        )
        assert result is mock_memori_cls.return_value

    @patch.dict("sys.modules", {"memori": MagicMock()})
    def test_get_memori_custom_ids(self):
        """Custom entity_id and process_id should be used."""
        import sys

        mock_module = sys.modules["memori"]
        mock_memori_cls = MagicMock()
        mock_module.Memori = mock_memori_cls

        mw = self._make_middleware(entity_id="custom_ent", process_id="custom_proc")
        mw._get_memori()

        mock_memori_cls.assert_called_once_with(
            entity_id="custom_ent",
            process_id="custom_proc",
            db_path=":memory:",
        )
