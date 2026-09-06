"""A6 shared resolvability gate across the three regional examination resolvers.

Task 7 (batch A6) — spec §5.3 A6: CN / EP / JP examination resolvers must not
blindly delegate a deterministically-unresolvable id (PCT / unsupported bare /
foreign shape) to the EPO ``lookup_family`` remote call.  Each resolver gains a
one-line ``verdict_of`` gate immediately before delegation, routing an
unresolvable id to its existing failure channel carrying generic
publication-format guidance (spec §5.4 template, reason_code
ERR_UNRESOLVABLE_ID).  resolvable / None / a translator exception keep the
legacy lookup_family delegation (fail-open contract, spec §7).

Local-number shapes still short-circuit to the resolver's existing direct path —
the mock EPO client must never be hit for them (zero regression).

Async resolvers are driven with ``asyncio.run`` in sync tests, the idiom used
elsewhere in this repo's long-task suites (cf. tests/test_patent_id_translator).
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from sources.long_task.china_examination import resolve_cn_application_number
from sources.long_task.epo_examination import resolve_ep_application_number
from sources.long_task.japan_examination import resolve_jp_application_number
from sources.long_task.patent_family import EPOError


def _awaited(coro):
    return asyncio.run(coro)


def _epo_client_error():
    """EPO client whose lookup_family raises a family EPOError (network class)."""
    client = AsyncMock()
    client.lookup_family = AsyncMock(side_effect=EPOError("boom"))
    return client


# ── CN resolver ────────────────────────────────────────────────────────────────


class TestCNGate:
    def test_local_cn_number_short_circuits_direct(self):
        epo = AsyncMock()  # must never be reached
        cn_app, ctx = _awaited(resolve_cn_application_number("CN201910887654.9", epo))
        assert ctx.get("direct_cn") is True
        assert cn_app.isdigit()
        epo.lookup_family.assert_not_awaited()

    @pytest.mark.parametrize("bad", [
        "PCTUS2021059064",      # PCT international application number
        "12345678901",           # unsupported bare foreign shape
        "EP09123456",            # non-CN office number
    ])
    def test_unresolvable_id_does_not_touch_epo_and_guides(self, bad):
        epo = _epo_client_error()
        with pytest.raises(ValueError) as exc:
            _awaited(resolve_cn_application_number(bad, epo))
        msg = str(exc.value)
        assert "公开号" in msg or "WO" in msg
        epo.lookup_family.assert_not_awaited()

    def test_translator_exception_falls_open_to_epo(self):
        # translator internal exception → old delegation unchanged (fail-open).
        epo = _epo_client_error()
        with patch(
            "sources.patent_id_translator.verdict_of",
            side_effect=ValueError("translator boom"),
        ):
            with pytest.raises(ValueError) as exc:
                _awaited(resolve_cn_application_number("PCTUS2021059064", epo))
        assert "EPO family lookup failed" in str(exc.value)
        epo.lookup_family.assert_awaited_once()

    def test_resolvable_foreign_id_still_delegates_to_epo(self):
        # a resolvable non-CN number (e.g. US grant) legitimately asks EPO for
        # its CN family member — delegation preserved (zero regression).
        epo = _epo_client_error()
        with pytest.raises(ValueError) as exc:
            _awaited(resolve_cn_application_number("US12506212", epo))
        assert "EPO family lookup failed" in str(exc.value)
        epo.lookup_family.assert_awaited_once()


# ── EP resolver ────────────────────────────────────────────────────────────────


class TestEPGate:
    def test_local_ep_number_short_circuits_direct(self):
        epo = AsyncMock()
        ep_app, ctx = _awaited(resolve_ep_application_number("EP09123456", epo))
        assert ctx.get("direct_ep") is True
        assert ep_app.isdigit()
        epo.lookup_family.assert_not_awaited()

    @pytest.mark.parametrize("bad", [
        "PCTUS2021059064",   # PCT international application number
        "DE102018123456",     # foreign office number (fails EP-local regex)
        "JP2019061234",       # non-EP office number
    ])
    def test_unresolvable_id_does_not_touch_epo_and_guides(self, bad):
        epo = _epo_client_error()
        with pytest.raises(ValueError) as exc:
            _awaited(resolve_ep_application_number(bad, epo))
        msg = str(exc.value)
        assert "公开号" in msg or "WO" in msg
        epo.lookup_family.assert_not_awaited()

    def test_translator_exception_falls_open_to_epo(self):
        epo = _epo_client_error()
        with patch(
            "sources.patent_id_translator.verdict_of",
            side_effect=ValueError("translator boom"),
        ):
            with pytest.raises(ValueError) as exc:
                _awaited(resolve_ep_application_number("PCTUS2021059064", epo))
        assert "EPO family lookup failed" in str(exc.value)
        epo.lookup_family.assert_awaited_once()

    def test_resolvable_foreign_id_still_delegates_to_epo(self):
        epo = _epo_client_error()
        with pytest.raises(ValueError) as exc:
            _awaited(resolve_ep_application_number("US12506212", epo))
        assert "EPO family lookup failed" in str(exc.value)
        epo.lookup_family.assert_awaited_once()


# ── JP resolver ─────────────────────────────────────────────────────────────────


class TestJPGate:
    def test_local_jp_number_short_circuits_direct(self):
        epo = AsyncMock()
        jp_app, ctx = _awaited(resolve_jp_application_number("JP2019-061234", epo))
        assert ctx.get("direct_jp") is True
        assert jp_app.isdigit()
        epo.lookup_family.assert_not_awaited()

    @pytest.mark.parametrize("bad", [
        "PCTUS2021059064",  # PCT international application number
        "EP09123456",         # non-JP office number
        "DE102018123456",     # foreign office number (fails JP-local regex)
    ])
    def test_unresolvable_id_does_not_touch_epo_and_guides(self, bad):
        epo = _epo_client_error()
        jp_app, ctx = _awaited(resolve_jp_application_number(bad, epo))
        assert jp_app is None
        assert ctx.get("unresolvable") is True
        assert "公开号" in str(ctx.get("error", "")) or "WO" in str(ctx.get("error", ""))
        epo.lookup_family.assert_not_awaited()

    def test_local_no_epo_client_returns_none(self):
        # epo_client=None still short-circuits before any lookup or gate.
        jp_app, ctx = _awaited(resolve_jp_application_number("JP2019-061234", None))
        assert jp_app is None
        assert ctx == {}

    def test_translator_exception_falls_open_to_epo(self):
        epo = _epo_client_error()
        with patch(
            "sources.patent_id_translator.verdict_of",
            side_effect=ValueError("translator boom"),
        ):
            jp_app, _ctx = _awaited(
                resolve_jp_application_number("PCTUS2021059064", epo))
        assert jp_app is None  # EPO path returns None on its own failure
        epo.lookup_family.assert_awaited_once()
