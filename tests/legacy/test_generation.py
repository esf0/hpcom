# from ssfm_gpu.propagation import create_channel_parameters
#
# from hpcom._legacy.signal import create_wdm_parameters, generate_wdm
#
#
# # create parameters
# wdm = create_wdm_parameters(n_channels=1, p_ave_dbm=6, n_symbols=2 ** 16, m_order=16, roll_off=0.1, upsampling=16,
#                             downsampling_rate=1, symb_freq=34e9, channel_spacing=75e9, n_polarisations=2)
#
#
# channel = create_channel_parameters(n_spans=25,
#                                     z_span=80,
#                                     alpha_db=0.2,
#                                     gamma=1.2,
#                                     noise_figure_db=4.5,
#                                     dispersion_parameter=16.8,
#                                     dz=1)

import pytest
import numpy as np
from unittest.mock import patch

# Assuming your functions are in a file named 'hpcom._legacy.modulation.py'
from hpcom._legacy.modulation import get_constellation_point, get_constellation
from hpcom._legacy.signal import generate_wdm


# =====================================================================
# Fixtures & Mocks
# =====================================================================

@pytest.fixture
def mock_constellation_dict():
    """Provides a predictable 2-bit (QPSK-like) constellation dictionary."""
    return {
        "00": 1 + 1j,
        "01": -1 + 1j,
        "10": -1 - 1j,
        "11": 1 - 1j
    }


@pytest.fixture
def mock_generate_base_return():
    """Mock return values for generate_wdm_base: (signal_temp, additional_dict)"""
    sig_temp = np.ones(100, dtype=complex)  # 100 samples
    add_dict = {
        'bits': '1010',
        'points': np.array([1 + 1j]),
        'ft_filter_values': np.array([0.5])
    }
    return sig_temp, add_dict


@pytest.fixture
def wdm_params():
    """Provides a baseline valid WDM configuration."""
    return {
        'symb_freq': 34e9,
        'upsampling': 16,
        'channel_spacing': 75e9,
        'n_channels': 3,
        'n_polarisations': 1,
        'p_ave': 2.0
    }


@pytest.fixture
def mock_base_return():
    """
    Returns a predictable output for generate_wdm_base.
    Signal is length 10.
    """
    sig_temp = np.ones(10, dtype=complex)
    add_dict = {
        'bits': '1010',
        'points': np.array([1 + 1j]),
        'ft_filter_values': np.array([0.5])
    }
    return sig_temp, add_dict


# Helper to bypass tensorflow GPU allocation in tests
def bypass_tf_cast(x, dtype):
    return x

# =====================================================================
# Tests for get_constellation_point
# =====================================================================

@patch('hpcom._legacy.modulation.get_n_bits')
def test_get_constellation_point_exact_bits(mock_get_n_bits, mock_constellation_dict):
    """Test chunking and mapping when bit_data length is a perfect multiple of n."""
    mock_get_n_bits.return_value = 2

    # "0011" should split perfectly into ["00", "11"] -> [1+1j, 1-1j]
    result = get_constellation_point("0011", type="qpsk", constellation=mock_constellation_dict)

    expected = np.array([1 + 1j, 1 - 1j])
    np.testing.assert_array_equal(result, expected)


@patch('hpcom._legacy.modulation.get_n_bits')
def test_get_constellation_point_padding(mock_get_n_bits, mock_constellation_dict):
    """Test that bit_data is left-padded with zeros if its length isn't a multiple of n."""
    mock_get_n_bits.return_value = 2

    # "110" has length 3. It should pad to 4 bits -> "0110"
    # "0110" splits into ["01", "10"] -> [-1+1j, -1-1j]
    result = get_constellation_point("110", type="qpsk", constellation=mock_constellation_dict)

    expected = np.array([-1 + 1j, -1 - 1j])
    np.testing.assert_array_equal(result, expected)


@patch('hpcom._legacy.modulation.generate_constellation_dict')
@patch('hpcom._legacy.modulation.get_n_bits')
def test_get_constellation_point_generation_fallback(mock_get_n_bits, mock_generate_dict, mock_constellation_dict):
    """Test that generate_constellation_dict is called if no constellation is provided."""
    mock_get_n_bits.return_value = 2
    mock_generate_dict.return_value = (mock_constellation_dict, None)

    # Passing constellation=None to trigger the internal generation
    result = get_constellation_point("00", type="qpsk", constellation=None)

    mock_generate_dict.assert_called_once_with(2)
    np.testing.assert_array_equal(result, np.array([1 + 1j]))


# =====================================================================
# Tests for get_constellation
# =====================================================================

@patch('hpcom._legacy.modulation.get_constellation_point')
@patch('hpcom._legacy.modulation.get_n_bits')
def test_get_constellation(mock_get_n_bits, mock_get_constellation_point):
    """Test that get_constellation maps binary strings sequentially."""
    mock_get_n_bits.return_value = 2  # 2^2 = 4 points total

    # Mocking the returns of get_constellation_point for loop indices 0, 1, 2, 3
    # Inside the loop, it converts i to binary: "0" -> "0", "1" -> "1", "2" -> "10", "3" -> "11"
    mock_get_constellation_point.side_effects = [
        np.array([10 + 0j]),  # for i=0 ("0")
        np.array([20 + 0j]),  # for i=1 ("1")
        np.array([30 + 0j]),  # for i=2 ("10")
        np.array([40 + 0j]),  # for i=3 ("11")
    ]
    mock_get_constellation_point.side_effect = mock_get_constellation_point.side_effects

    result = get_constellation("qpsk")

    expected = np.array([10 + 0j, 20 + 0j, 30 + 0j, 40 + 0j], dtype=complex)
    np.testing.assert_array_equal(result, expected)

    # Verify it called the inner function with the right arguments
    assert mock_get_constellation_point.call_count == 4
    mock_get_constellation_point.assert_any_call("0", "qpsk")
    mock_get_constellation_point.assert_any_call("11", "qpsk")


# =====================================================================
# Tests
# =====================================================================

@patch('hpcom._legacy.signal.check_wdm_parameters')
def test_invalid_wdm_parameters(mock_check):
    """Test that function aborts if parameter check fails."""
    mock_check.return_value = 1  # Simulate an error

    # Passing an empty dict because it shouldn't get past the check anyway
    result = generate_wdm({})

    assert result == -1
    mock_check.assert_called_once()


@patch('hpcom._legacy.signal.tf.cast')
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters')
def test_single_polarization_no_inputs(mock_check, mock_gen_base, mock_tf_cast,
                                       wdm_params, mock_generate_base_return):
    """Test 1 Pol, No Bits/Points/Filters provided (generated from scratch)."""
    mock_check.return_value = 0
    mock_gen_base.return_value = mock_generate_base_return

    # Let tf.cast just return the input array so we can verify the math
    mock_tf_cast.side_effect = lambda x, dtype: x

    signal, additional = generate_wdm(wdm_params)

    # Verify generate_wdm_base was called exactly n_channels times
    assert mock_gen_base.call_count == 3

    # Verify outputs
    assert len(signal) == 100  # Original signal temp was 100 length
    assert len(additional['bits_x']) == 3
    assert len(additional['points_x']) == 3
    assert len(additional['ft_filter_values_x']) == 3

    # Verify nothing went into y-pol
    assert additional['bits_y'] == []
    assert additional['points_y'] == []
    assert additional['ft_filter_values_y'] == []


@patch('hpcom._legacy.signal.tf.cast')
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters')
def test_dual_polarization_with_points_and_filters(mock_check, mock_gen_base, mock_tf_cast,
                                                   wdm_params, mock_generate_base_return):
    """Test 2 Pols, with pre-existing points and ft_filter_values."""
    mock_check.return_value = 0
    mock_gen_base.return_value = mock_generate_base_return
    mock_tf_cast.side_effect = lambda x, dtype: x

    # Update params to 2 polarisations
    wdm_params['n_polarisations'] = 2

    # Create mock inputs matching n_channels=3
    mock_points_x = [np.array([1]), np.array([2]), np.array([3])]
    mock_points_y = [np.array([4]), np.array([5]), np.array([6])]
    mock_points = (mock_points_x, mock_points_y)

    mock_filters_x = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
    mock_filters_y = [np.array([0.4]), np.array([0.5]), np.array([0.6])]
    mock_filters = (mock_filters_x, mock_filters_y)

    sig_x, sig_y, additional = generate_wdm(wdm_params, points=mock_points, ft_filter_values=mock_filters)

    # generate_wdm_base should be called 6 times (3 channels * 2 pols)
    assert mock_gen_base.call_count == 6

    # Because 'points' were passed IN, they shouldn't be extracted from the additional return dict.
    # The function appends them to points_x/y only if points is None.
    # Therefore, points_x in the output dict should equal the input points_x (by reference)
    # Wait, looking at the code: if points is NOT None, points_x is populated at the start,
    # but NOT appended to in the loop. Let's verify the list stays exactly as passed.
    assert additional['points_x'] == mock_points_x
    assert additional['points_y'] == mock_points_y

    assert len(sig_x) == 100
    assert len(sig_y) == 100


@patch('hpcom._legacy.signal.tf.cast')
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters')
def test_single_polarization_with_bits(mock_check, mock_gen_base, mock_tf_cast,
                                       wdm_params, mock_generate_base_return):
    """Test 1 Pol, with pre-existing bits."""
    mock_check.return_value = 0
    mock_gen_base.return_value = mock_generate_base_return
    mock_tf_cast.side_effect = lambda x, dtype: x

    mock_bits_x = ["0101", "1100", "0011"]  # 3 channels
    mock_bits = (mock_bits_x,)  # Tuple containing X pol bits

    signal, additional = generate_wdm(wdm_params, bits=mock_bits)

    # The function should pass the respective bit strings down to generate_wdm_base
    # Let's inspect the first call to generate_wdm_base to ensure bits were passed
    first_call_kwargs = mock_gen_base.call_args_list[0].kwargs
    assert first_call_kwargs['bits'] == "0101"

    assert additional['bits_x'] == mock_bits_x


# =====================================================================
# Tests: Parameter Validation
# =====================================================================

@patch('hpcom._legacy.signal.check_wdm_parameters')
def test_invalid_wdm_parameters(mock_check):
    """Test that the function aborts if parameter check fails."""
    mock_check.return_value = 1
    result = generate_wdm({})
    assert result == -1


# =====================================================================
# Tests: 1 Polarization Branches
# =====================================================================

@patch('hpcom._legacy.signal.tf.cast', side_effect=bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_1pol_no_inputs_multiple_channels(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """Test 1 Pol, No inputs provided, n_channels > 1. Hits the 'else' generation branch."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 3
    wdm_params['n_polarisations'] = 1

    signal, additional = generate_wdm(wdm_params)

    assert mock_base.call_count == 3
    # Check that it called base WITHOUT bits or points
    mock_base.assert_called_with(wdm_params, seed=2, ft_filter_values=None)

    assert len(signal) == 10
    assert len(additional['bits_x']) == 3
    assert additional['bits_y'] == []  # Y pol should be empty
    assert additional['points_y'] == []


@patch('hpcom._legacy.signal.tf.cast', side_effect=bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_1pol_with_bits_single_channel(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """Test 1 Pol, Passing bits, n_channels = 1. Bypasses the signal accumulation block."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 1
    wdm_params['n_polarisations'] = 1

    mock_bits = (["0101"],)  # Tuple of lists for X pol

    signal, additional = generate_wdm(wdm_params, bits=mock_bits)

    assert mock_base.call_count == 1
    # Check that bits were passed to the base function
    mock_base.assert_called_with(wdm_params, bits="0101", seed=0, ft_filter_values=None)

    # Since bits were passed IN, they should be in the output
    assert additional['bits_x'] == ["0101"]


@patch('hpcom._legacy.signal.tf.cast', side_effect=bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_1pol_with_points_and_filters(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """Test 1 Pol, Passing points AND filters."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 2

    mock_points = ([np.array([1j]), np.array([-1j])],)
    mock_filters = ([np.array([0.1]), np.array([0.2])],)

    signal, additional = generate_wdm(wdm_params, points=mock_points, ft_filter_values=mock_filters)

    assert mock_base.call_count == 2
    # Verify the second channel call had the correct points and filters applied
    mock_base.assert_called_with(wdm_params, points=mock_points[0][1], seed=1, ft_filter_values=mock_filters[0][1])


# =====================================================================
# Tests: 2 Polarization Branches
# =====================================================================

@patch('hpcom._legacy.signal.tf.cast', side_effect=bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_2pol_no_inputs_p_ave_halved(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """Test 2 Pols, no inputs. Ensures power is halved and 3 items are returned."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 2
    wdm_params['n_polarisations'] = 2
    wdm_params['p_ave'] = 4.0

    # Should return signal_x, signal_y, and additional
    sig_x, sig_y, additional = generate_wdm(wdm_params)

    # Base should be called 4 times (2 channels * 2 pols)
    assert mock_base.call_count == 4

    # Verify the dictionary passed to base had its power halved
    called_wdm_dict = mock_base.call_args[0][0]
    assert called_wdm_dict['p_ave'] == 2.0

    # Ensure all lists are populated
    assert len(additional['bits_x']) == 2
    assert len(additional['bits_y']) == 2


@patch('hpcom._legacy.signal.tf.cast', side_effect=bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_2pol_with_bits(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """Test 2 Pols, passing bits. Ensures X and Y pols get correct bit streams."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 1
    wdm_params['n_polarisations'] = 2

    mock_bits = (["1111"], ["0000"])  # X bits and Y bits

    sig_x, sig_y, additional = generate_wdm(wdm_params, bits=mock_bits)

    # First call is for X polarization
    first_call_kwargs = mock_base.call_args_list[0].kwargs
    assert first_call_kwargs['bits'] == "1111"

    # Second call is for Y polarization
    second_call_kwargs = mock_base.call_args_list[1].kwargs
    assert second_call_kwargs['bits'] == "0000"


@patch('hpcom._legacy.signal.tf.cast', side_effect=bypass_tf_cast)
@patch('hpcom._legacy.signal.generate_wdm_base')
@patch('hpcom._legacy.signal.check_wdm_parameters', return_value=0)
def test_2pol_with_points_and_filters(mock_check, mock_base, mock_tf, wdm_params, mock_base_return):
    """Test 2 Pols, passing both points and filters. The ultimate stress test of unpacking."""
    mock_base.return_value = mock_base_return
    wdm_params['n_channels'] = 2
    wdm_params['n_polarisations'] = 2

    mock_points = (
        [np.array([1j]), np.array([2j])],  # X points for ch0, ch1
        [np.array([-1j]), np.array([-2j])]  # Y points for ch0, ch1
    )
    mock_filters = (
        [np.array([0.1]), np.array([0.2])],  # X filters for ch0, ch1
        [np.array([0.3]), np.array([0.4])]  # Y filters for ch0, ch1
    )

    sig_x, sig_y, additional = generate_wdm(wdm_params, points=mock_points, ft_filter_values=mock_filters)

    # Inspect the final Y polarization call (channel 1, Y pol)
    last_call_kwargs = mock_base.call_args_list[-1].kwargs

    # Verify the points unpacked correctly
    np.testing.assert_array_equal(last_call_kwargs['points'], np.array([-2j]))

    # Verify the filters unpacked correctly
    np.testing.assert_array_equal(last_call_kwargs['ft_filter_values'], np.array([0.4]))

    # Verify seeds offset correctly (seed=wdm_index + wdm['n_channels'])
    # For channel 1, Y pol, seed should be 1 + 2 = 3
    assert last_call_kwargs['seed'] == 3