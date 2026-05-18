"""Structural unit tests for legacy modulation/signal orchestration.

These tests mock out the inner numeric kernels (generate_wdm_base, tf.cast,
check_wdm_parameters) and verify the *plumbing* of generate_wdm — call counts,
which kwargs flow to the base function, how the n_polarisations and
points/bits/filter inputs change the dispatch. They do not validate physics;
numerical correctness belongs to the Path-B parity tests against v1.
"""

import pytest
import numpy as np
from unittest.mock import patch

from hpcom._legacy.modulation import get_constellation_point, get_constellation
from hpcom._legacy.signal import generate_wdm


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_constellation_dict():
    """Predictable 2-bit (QPSK-like) constellation dictionary."""
    return {
        "00": 1 + 1j,
        "01": -1 + 1j,
        "10": -1 - 1j,
        "11": 1 - 1j,
    }


@pytest.fixture(params=[10, 100], ids=["short", "long"])
def mock_base_return(request):
    """Predictable (signal_temp, additional_dict) for generate_wdm_base."""
    sig_temp = np.ones(request.param, dtype=complex)
    add_dict = {
        'bits': '1010',
        'points': np.array([1 + 1j]),
        'ft_filter_values': np.array([0.5]),
    }
    return sig_temp, add_dict


@pytest.fixture
def wdm_params():
    """Baseline valid WDM configuration."""
    return {
        'symb_freq': 34e9,
        'upsampling': 16,
        'channel_spacing': 75e9,
        'n_channels': 3,
        'n_polarisations': 1,
        'p_ave': 2.0,
    }


def _bypass_tf_cast(x, dtype):
    """tf.cast stub that returns the input unchanged (no GPU alloc in tests)."""
    return x


# =====================================================================
# get_constellation_point
# =====================================================================

@patch('hpcom._legacy.modulation.get_n_bits')
def test_get_constellation_point_exact_bits(mock_get_n_bits, mock_constellation_dict):
    """Chunking when bit_data length is a perfect multiple of n."""
    mock_get_n_bits.return_value = 2

    # "0011" splits into ["00", "11"] -> [1+1j, 1-1j]
    result = get_constellation_point("0011", type="qpsk", constellation=mock_constellation_dict)

    np.testing.assert_array_equal(result, np.array([1 + 1j, 1 - 1j]))


@patch('hpcom._legacy.modulation.get_n_bits')
def test_get_constellation_point_padding(mock_get_n_bits, mock_constellation_dict):
    """bit_data is left-padded with zeros when length isn't a multiple of n."""
    mock_get_n_bits.return_value = 2

    # "110" (length 3) pads to "0110" -> ["01", "10"] -> [-1+1j, -1-1j]
    result = get_constellation_point("110", type="qpsk", constellation=mock_constellation_dict)

    np.testing.assert_array_equal(result, np.array([-1 + 1j, -1 - 1j]))


@patch('hpcom._legacy.modulation.generate_constellation_dict')
@patch('hpcom._legacy.modulation.get_n_bits')
def test_get_constellation_point_generation_fallback(mock_get_n_bits, mock_generate_dict, mock_constellation_dict):
    """generate_constellation_dict is called when no constellation is provided."""
    mock_get_n_bits.return_value = 2
    mock_generate_dict.return_value = (mock_constellation_dict, None)

    result = get_constellation_point("00", type="qpsk", constellation=None)

    mock_generate_dict.assert_called_once_with(2)
    np.testing.assert_array_equal(result, np.array([1 + 1j]))


# =====================================================================
# get_constellation
# =====================================================================

@patch('hpcom._legacy.modulation.get_constellation_point')
@patch('hpcom._legacy.modulation.get_n_bits')
def test_get_constellation(mock_get_n_bits, mock_get_constellation_point):
    """get_constellation iterates i=0..2^n-1, formats i as plain binary, and
    stores the first element of each returned array."""
    mock_get_n_bits.return_value = 2  # 2^2 = 4 points total

    # Production loop yields "{0:b}".format(i) -> "0", "1", "10", "11"
    # The fix in modulation.py indexes [0] on each returned array.
    mock_get_constellation_point.side_effect = [
        np.array([10 + 0j]),  # i=0 -> "0"
        np.array([20 + 0j]),  # i=1 -> "1"
        np.array([30 + 0j]),  # i=2 -> "10"
        np.array([40 + 0j]),  # i=3 -> "11"
    ]

    result = get_constellation("qpsk")

    expected = np.array([10 + 0j, 20 + 0j, 30 + 0j, 40 + 0j], dtype=complex)
    np.testing.assert_array_equal(result, expected)

    assert mock_get_constellation_point.call_count == 4
    for bits in ("0", "1", "10", "11"):
        mock_get_constellation_point.assert_any_call(bits, "qpsk")


# =====================================================================
# generate_wdm — parameter validation
# =====================================================================

@patch('hpcom._legacy.signal.check_wdm_parameters')
def test_invalid_wdm_parameters(mock_check):
    """generate_wdm aborts with -1 if check_wdm_parameters reports an error."""
    mock_check.return_value = 1
    result = generate_wdm({})
    assert result == -1
    mock_check.assert_called_once()


# =====================================================================
# generate_wdm — 1 polarisation
# =====================================================================

@patch('hpcom._legacy.signal.tf.cast', side_effect=_bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_1pol_no_inputs_multiple_channels(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """1 pol, no bits/points/filters: hits the from-scratch generation branch."""
    mock_base.return_value = mock_base_return
    sig_len = len(mock_base_return[0])

    signal, additional = generate_wdm(wdm_params)

    assert mock_base.call_count == 3
    mock_base.assert_called_with(wdm_params, seed=2, ft_filter_values=None)
    assert len(signal) == sig_len
    assert len(additional['bits_x']) == 3
    assert len(additional['points_x']) == 3
    assert len(additional['ft_filter_values_x']) == 3
    assert additional['bits_y'] == []
    assert additional['points_y'] == []
    assert additional['ft_filter_values_y'] == []


@patch('hpcom._legacy.signal.tf.cast', side_effect=_bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_1pol_with_bits_single_channel(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """1 pol, n_channels=1, bits supplied: skips the accumulation branch."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 1

    signal, additional = generate_wdm(wdm_params, bits=(["0101"],))

    assert mock_base.call_count == 1
    mock_base.assert_called_with(wdm_params, bits="0101", seed=0, ft_filter_values=None)
    assert additional['bits_x'] == ["0101"]


@patch('hpcom._legacy.signal.tf.cast', side_effect=_bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_1pol_with_points_and_filters(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """1 pol, points + ft_filter_values supplied: indexes per channel."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 2

    mock_points = ([np.array([1j]), np.array([-1j])],)
    mock_filters = ([np.array([0.1]), np.array([0.2])],)

    generate_wdm(wdm_params, points=mock_points, ft_filter_values=mock_filters)

    assert mock_base.call_count == 2
    mock_base.assert_called_with(wdm_params, points=mock_points[0][1], seed=1, ft_filter_values=mock_filters[0][1])


# =====================================================================
# generate_wdm — 2 polarisations
# =====================================================================

@patch('hpcom._legacy.signal.tf.cast', side_effect=_bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_2pol_no_inputs_p_ave_halved(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """2 pols, no inputs: p_ave is halved and the function returns 3-tuple."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 2
    wdm_params['n_polarisations'] = 2
    wdm_params['p_ave'] = 4.0

    sig_x, sig_y, additional = generate_wdm(wdm_params)

    assert mock_base.call_count == 4  # 2 channels * 2 pols
    called_wdm_dict = mock_base.call_args[0][0]
    assert called_wdm_dict['p_ave'] == 2.0
    assert len(additional['bits_x']) == 2
    assert len(additional['bits_y']) == 2


@patch('hpcom._legacy.signal.tf.cast', side_effect=_bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_2pol_with_bits(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """2 pols, bits supplied: X and Y bit streams reach generate_wdm_base."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 1
    wdm_params['n_polarisations'] = 2

    generate_wdm(wdm_params, bits=(["1111"], ["0000"]))

    assert mock_base.call_args_list[0].kwargs['bits'] == "1111"
    assert mock_base.call_args_list[1].kwargs['bits'] == "0000"


@patch('hpcom._legacy.signal.tf.cast', side_effect=_bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_2pol_with_points_and_filters(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """2 pols, points + filters supplied: full nested unpacking and Y-pol seed offset."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 2
    wdm_params['n_polarisations'] = 2

    mock_points = (
        [np.array([1j]), np.array([2j])],
        [np.array([-1j]), np.array([-2j])],
    )
    mock_filters = (
        [np.array([0.1]), np.array([0.2])],
        [np.array([0.3]), np.array([0.4])],
    )

    generate_wdm(wdm_params, points=mock_points, ft_filter_values=mock_filters)

    # Final call is channel 1, Y pol: seed = wdm_index + n_channels = 1 + 2 = 3
    last = mock_base.call_args_list[-1].kwargs
    np.testing.assert_array_equal(last['points'], np.array([-2j]))
    np.testing.assert_array_equal(last['ft_filter_values'], np.array([0.4]))
    assert last['seed'] == 3


@patch('hpcom._legacy.signal.tf.cast', side_effect=_bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_2pol_with_points_input_lists_preserved(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """2 pols with points supplied: caller's points lists are not re-appended to."""
    mock_base.return_value = mock_base_return
    wdm_params['n_polarisations'] = 2

    mock_points_x = [np.array([1]), np.array([2]), np.array([3])]
    mock_points_y = [np.array([4]), np.array([5]), np.array([6])]
    mock_filters_x = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
    mock_filters_y = [np.array([0.4]), np.array([0.5]), np.array([0.6])]

    sig_x, sig_y, additional = generate_wdm(
        wdm_params,
        points=(mock_points_x, mock_points_y),
        ft_filter_values=(mock_filters_x, mock_filters_y),
    )

    assert mock_base.call_count == 6
    assert additional['points_x'] == mock_points_x
    assert additional['points_y'] == mock_points_y
    assert len(sig_x) == len(mock_base_return[0])
    assert len(sig_y) == len(mock_base_return[0])
