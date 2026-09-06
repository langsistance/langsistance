"""A6 shared resolvability gate across the three regional examination resolvers.

Task 7 (batch A6) — spec §5.3 A6: CN / EP / JP examination resolvers must not
blindly delegate an id the EPO ``lookup_family`` call could never use to the
remote API.  Each resolver gains a one-line ``unresolvable_gate_error`` gate
immediately before delegation, routing it to its existing failure channel
carrying generic publication-format guidance (spec §5.4 template, reason_code
ERR_UNRESOLVABLE_ID) — but the gate is *fail-open* by design (review F2 · T7).

Gate boundary (see translator ``unresolvable_gate_error``): only a **PCT**
international number (``id_type=pct``) and a **bare unsupported** shape with no
recognisable office country (bare ≥9-digit run / gibberish) are guided.  A
non-local id that still carries a *recognisable office* country — EP / JP / DE /
GB / FR / … — legitimately delegates to EPO ``lookup_family`` (EPO OPS accepts
these foreign docdb ids and returns their family members for the calling
office).  resolvable / None / a translator exception likewise keep the legacy
lookup_family delegation (fail-open contract, spec §7).

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
        "PCTUS2021059064",      # PCT international application number (id_type=pct)
        "2021059064",           # bare ≥9-digit run, no office prefix (unsupported)
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

    @pytest.mark.parametrize("foreign", [
        "WO2021059064",  # WO publication → EPO cross-office (WO→CN positive)
        "EP09123456",    # recognisable office prefix EP, non-CN → delegate
        "DE102018123456",  # recognisable office prefix DE, non-CN → delegate
    ])
    def test_foreign_office_number_delegates_to_epo(self, foreign):
        # F2 (T7): a number carrying a *recognisable office* country prefix is
        # cross-office delegable — EPO OPS accepts these docdb ids.  Only pct /
        # bare-gibberish are gated.
        epo = _epo_client_error()
        with pytest.raises(ValueError) as exc:
            _awaited(resolve_cn_application_number(foreign, epo))
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
        # EP's local short-circuit regex is prefix-anchored (not full-string)
        # so a bare digit run always resolves "direct" before the gate — only a
        # PCT prefix (id_type=pct, letters first) reaches the gate here.  The
        # bare-≥9-digit guide path is exercised on the CN resolver (digits fall
        # through to the gate there) and directly on the translator helper.
        "PCTUS2021059064",   # PCT international application number (id_type=pct)
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

    @pytest.mark.parametrize("foreign", [
        "WO2021059064",    # WO publication → EPO cross-office delegation
        "JP2019061234",    # recognisable office prefix JP, non-EP → delegate
        "DE102018123456",  # recognisable office prefix DE, non-EP → delegate
    ])
    def test_foreign_office_number_delegates_to_epo(self, foreign):
        # F2 (T7): recognisable-office ids cross-office delegate (not gated).
        epo = _epo_client_error()
        with pytest.raises(ValueError) as exc:
            _awaited(resolve_ep_application_number(foreign, epo))
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
        # JP's local short-circuit regex is prefix-anchored (not full-string)
        # so a bare digit run always resolves "direct" before the gate — only a
        # PCT prefix (id_type=pct, letters first) reaches the gate here.  The
        # bare-≥9-digit guide path is exercised on the CN resolver (digits fall
        # through to the gate there) and directly on the translator helper.
        "PCTUS2021059064",  # PCT international application number (id_type=pct)
    ])
    def test_unresolvable_id_does_not_touch_epo_and_guides(self, bad):
        # JP resolver relays an unresolvable via its (None, ctx) channel.
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

    @pytest.mark.parametrize("foreign", [
        "WO2021059064",    # WO publication → EPO cross-office delegation
        "US12506212",      # US→JP delegation positive (resolvable grant)
        "EP09123456",      # recognisable office prefix EP, non-JP → delegate
        "DE102018123456",  # recognisable office prefix DE, non-JP → delegate
    ])
    def test_foreign_office_number_delegates_to_epo(self, foreign):
        # F2 (T7): recognisable-office / resolvable ids cross-office delegate;
        # lookup_family runs and returns None on its own EPOError (no gate).
        epo = _epo_client_error()
        jp_app, _ctx = _awaited(
            resolve_jp_application_number(foreign, epo))
        assert jp_app is None  # EPO delegation path absorbed the error
        epo.lookup_family.assert_awaited_once()
