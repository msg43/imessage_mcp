"""The `search_page:` config section: listeners are loopback or
local-network IP literals only, ports never collide, paths stay under
data_root, and everything is off by default."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from imsg.config.schema import Config
from imsg.search_page.config import SearchPageConfig, parse_listen_address
from imsg.search_page.errors import SearchPageStartupError
from imsg.search_page.server import check_ports


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("127.0.0.1:8710", ("127.0.0.1", 8710)),
        ("192.168.1.20:8710", ("192.168.1.20", 8710)),
        ("10.0.0.5:9000", ("10.0.0.5", 9000)),
        ("172.16.4.4:8710", ("172.16.4.4", 8710)),
        ("[::1]:8710", ("::1", 8710)),
        ("[fe80::1]:8710", ("fe80::1", 8710)),
    ],
)
def test_loopback_and_local_network_addresses_are_accepted(entry: str, expected: tuple[str, int]) -> None:
    assert parse_listen_address(entry) == expected


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        ("0.0.0.0:8710", "wildcard"),
        ("[::]:8710", "wildcard"),
        ("100.101.102.103:8710", "Tailscale"),
        ("8.8.8.8:8710", "not a loopback or local-network"),
        ("mini.local:8710", "IP literal"),
        ("127.0.0.1", "address:port"),
        ("127.0.0.1:0", "valid port"),
        ("224.0.0.1:8710", "multicast"),
    ],
)
def test_other_listen_addresses_are_refused(entry: str, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        parse_listen_address(entry)


def test_defaults_are_off_and_loopback_only() -> None:
    cfg = SearchPageConfig()
    assert cfg.enabled is False
    assert cfg.model_api.enabled is False
    assert cfg.listen_addresses() == [("127.0.0.1", 8710)]


def test_model_api_port_may_not_equal_a_page_port() -> None:
    with pytest.raises(ValidationError, match="collides"):
        SearchPageConfig(listen=["127.0.0.1:8711"], model_api={"port": 8711})


@pytest.mark.parametrize("path", ["/etc/passwd", "~/secret", "../outside", "private/../../x"])
def test_paths_must_stay_under_data_root(path: str) -> None:
    with pytest.raises(ValidationError):
        SearchPageConfig(password_file=path)


def test_https_hosts_must_be_allowed_hosts() -> None:
    with pytest.raises(ValidationError, match="https_hosts"):
        SearchPageConfig(allowed_hosts=["127.0.0.1:8710"], https_hosts=["mini.example.ts.net"])


def test_host_entries_are_exact_values() -> None:
    with pytest.raises(ValidationError):
        SearchPageConfig(allowed_hosts=["http://mini:8710/"])
    with pytest.raises(ValidationError):
        SearchPageConfig(allowed_hosts=["*.local"])
    cfg = SearchPageConfig(allowed_hosts=["Mini.Local:8710"])
    assert cfg.allowed_hosts == ["mini.local:8710"]


def test_enabled_needs_allowed_hosts() -> None:
    with pytest.raises(ValidationError, match="allowed_hosts"):
        SearchPageConfig(enabled=True, allowed_hosts=[])


def test_full_config_loads_with_and_without_the_section(config_dict_factory: Any) -> None:
    assert Config.model_validate(config_dict_factory()).search_page.enabled is False
    raw = config_dict_factory()
    raw["search_page"] = {
        "enabled": True,
        "listen": ["127.0.0.1:8710", "192.168.1.20:8710"],
        "allowed_hosts": ["127.0.0.1:8710", "192.168.1.20:8710"],
        "model_api": {"enabled": True, "port": 8711},
    }
    cfg = Config.model_validate(raw)
    assert cfg.search_page.listen_addresses()[1] == ("192.168.1.20", 8710)
    raw["search_page"]["unknown_key"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_startup_refuses_the_public_mcp_port(config_dict_factory: Any) -> None:
    raw = config_dict_factory()
    raw["search_page"] = {"enabled": True, "listen": ["127.0.0.1:8700"], "allowed_hosts": ["127.0.0.1:8700"]}
    cfg = Config.model_validate(raw)
    with pytest.raises(SearchPageStartupError, match="port"):
        check_ports(cfg)
