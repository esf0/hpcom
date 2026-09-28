"""The HpcomModel contract, checked on every public configuration model.

Every model must: reject unknown keywords, be frozen (also through its
arrays), compare/hash by value (arrays included), re-validate on
``replace``/``model_copy(update=)``, and round-trip through JSON — including
polymorphic fields.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import (
    EDFA,
    ConstantCompositionDM,
    ConstellationConfig,
    Fiber,
    Link,
    OFDMConfig,
    OFDMReceiver,
    OFDMTransmitter,
    PilotConfig,
    Simulation,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMReceiver,
    WDMTransmitter,
)
from hpcom._model import HpcomModel
from hpcom.experimental import ExperimentalReceiver, ExperimentalTransmission
from hpcom.modulation import apsk, maxwell_boltzmann

_QAM = ConstellationConfig.from_square_qam(16)
_PMF = maxwell_boltzmann(_QAM.points, 0.1)
_PS = _QAM.with_probabilities(_PMF)
_DM = ConstantCompositionDM(target_pmf=_PMF, block_length=64)
_WDM = WDMConfig(n_channels=3, spacing_hz=50e9)
_OFDM = OFDMConfig(
    n_subcarriers=64, cp_len=8, n_guard=2, pilot_subcarrier_indices=(4, 20), oversampling=2
)
_FIBER = Fiber.ssmf(80.0, n_steps=10)
_LINK = Link.from_spans(2, _FIBER, EDFA(gain_db=16.0))

MODELS = {
    "ConstellationConfig": _PS,
    "apsk": apsk((4, 12), (1.0, 2.8)),
    "ConstantCompositionDM": _DM,
    "PilotConfig": PilotConfig(period=8, symbol=0.5 + 0.5j),
    "WDMConfig": _WDM,
    "OFDMConfig": _OFDM,
    "Fiber": _FIBER,
    "EDFA": EDFA(gain_db=16.0, noise_figure_db=4.5),
    "Link": _LINK,
    "SingleCarrierTransmitter": SingleCarrierTransmitter(
        constellation=_PS, dm=_DM, symbol_rate_hz=32e9, n_symbols=256, seed=1
    ),
    "SingleCarrierReceiver": SingleCarrierReceiver(
        constellation=_PS, dm=_DM, cd_compensation_s2=1e-22
    ),
    "WDMTransmitter": WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=256,
        samples_per_symbol=8,
        wdm=_WDM,
        polarization="dual",
        launch_power_total_dbm=5.0,
        pilots=PilotConfig(period=16),
    ),
    "WDMReceiver": WDMReceiver(
        modulation_order=16, samples_per_symbol=8, wdm=_WDM, channels=(2, 0)
    ),
    "OFDMTransmitter": OFDMTransmitter(
        modulation_order=4, subcarrier_spacing_hz=1e9, n_frames=4, ofdm=_OFDM
    ),
    "OFDMReceiver": OFDMReceiver(modulation_order=4, ofdm=_OFDM, equalize="pilot"),
    "ExperimentalTransmission": ExperimentalTransmission(
        modulation_order=4, symbol_rate_hz=1e9, n_symbols=16
    ),
    "ExperimentalReceiver": ExperimentalReceiver(modulation_order=4),
    "Simulation": Simulation(
        transmitter=SingleCarrierTransmitter(
            modulation_order=16, symbol_rate_hz=32e9, n_symbols=256
        ),
        link=_LINK,
        receiver=SingleCarrierReceiver(modulation_order=16),
        seed=3,
    ),
}


@pytest.fixture(params=list(MODELS), ids=list(MODELS))
def model(request) -> HpcomModel:
    return MODELS[request.param]


def test_unknown_keyword_is_rejected(model):
    data = {name: getattr(model, name) for name in type(model).model_fields}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(model)(**data, definitely_not_a_field=1)


def test_fields_cannot_be_reassigned(model):
    name = next(iter(type(model).model_fields))
    with pytest.raises(ValidationError, match="frozen"):
        setattr(model, name, getattr(model, name))


def test_array_fields_are_read_only(model):
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError, match="read-only"):
                value[0] = value[0]


def test_equal_to_a_rebuilt_copy_and_hashable(model):
    clone = type(model).model_validate({n: getattr(model, n) for n in type(model).model_fields})
    assert clone == model
    assert hash(clone) == hash(model)
    assert model != object()


def test_json_round_trip_restores_an_equal_model(model):
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model
    assert type(restored) is type(model)


def test_replace_revalidates():
    tx = MODELS["SingleCarrierTransmitter"]
    with pytest.raises(ValidationError, match=r"multiple of dm\.block_length"):
        tx.replace(n_symbols=100)
    with pytest.raises(ValidationError, match=r"multiple of dm\.block_length"):
        tx.model_copy(update={"n_symbols": 100})
    assert tx.replace(n_symbols=128).n_symbols == 128


def test_replace_resets_the_other_members_of_an_exclusive_group():
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=64)
    switched = tx.replace(constellation=apsk((4, 12), (1.0, 2.8)))
    assert switched.modulation_order is None
    assert switched.effective_modulation_order == 16
    assert switched.replace(modulation_order=64).constellation is None


def test_replace_rejects_unknown_fields():
    with pytest.raises(TypeError, match="has no field"):
        MODELS["Fiber"].replace(lenght_km=3.0)


def test_derived_state_is_never_stale_after_replace():
    """Pre-hardening, model_copy kept a cached 16-QAM constellation after switching to 64."""
    tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=64)
    assert tx.model_copy(update={"modulation_order": 64}).effective_modulation_order == 64
    dm = ConstantCompositionDM(target_pmf=_PMF, block_length=64)
    assert dm.replace(block_length=128).composition.sum() == 128


def test_polymorphic_fields_keep_their_concrete_type_through_json():
    sim = Simulation(
        transmitter=MODELS["WDMTransmitter"].replace(pilots=None, polarization="single"),
        link=Link(elements=(_FIBER, EDFA(gain_db=16.0), _FIBER)),
        receiver=MODELS["WDMReceiver"],
    )
    back = Simulation.model_validate_json(sim.model_dump_json())
    assert isinstance(back.transmitter, WDMTransmitter)
    assert isinstance(back.receiver, WDMReceiver)
    assert [type(e) for e in back.link.elements] == [Fiber, EDFA, Fiber]


def test_json_does_not_resolve_unregistered_types():
    payload = MODELS["Link"].model_dump(mode="json")
    payload["elements"][0]["type"] = "os.system"
    with pytest.raises(ValidationError, match=r"unknown model type 'os\.system'"):
        Link.model_validate(payload)


def test_constellation_equality_compares_arrays_by_value():
    """Pre-hardening this raised 'truth value of an array is ambiguous'."""
    a = ConstellationConfig.from_points(_QAM.points, _QAM.bit_labels)
    assert a == ConstellationConfig.from_square_qam(16)
    assert a != ConstellationConfig.from_square_qam(64)
    assert a != _PS
