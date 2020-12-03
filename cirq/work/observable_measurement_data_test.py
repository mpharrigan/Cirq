# Copyright 2020 The Cirq developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime

import numpy as np
import pytest

import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
    _check_and_get_real_coef,
    _obs_vals_from_measurements,
    _stats_from_measurements,
)
from cirq.work.observable_settings import _MeasurementSpec


def test_get_real_coef():
    q0 = cirq.LineQubit(0)
    assert _check_and_get_real_coef(cirq.Z(q0) * 2, atol=1e-8) == 2
    assert _check_and_get_real_coef(cirq.Z(q0) * complex(2.0), atol=1e-8) == 2
    with pytest.raises(ValueError):
        _check_and_get_real_coef(cirq.Z(q0) * 2.0j, atol=1e-8)


def test_obs_vals_from_measurements():
    bitstrings = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    vals = _obs_vals_from_measurements(bitstrings, qubit_to_index, obs, atol=1e-8)
    should_be = [10, -10, -10, 10]
    np.testing.assert_equal(vals, should_be)


def test_stats_from_measurements():
    bitstrings = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    mean, err = _stats_from_measurements(bitstrings, qubit_to_index, obs, atol=1e-8)

    # The mean is zero since our bitstrings have balanced even- and odd-
    # partiy cases.
    assert mean == 0

    # Since we multiplied our observable by 10, the standard deviation is
    # 10 [each obs val deviates by 10]. The variance is 10**2 and the
    # squared-standard-error-of-the-mean can be found by dividing by the
    # number of samples minus 1.
    assert err == 10 ** 2 / (4 - 1)


def test_observable_measured_result():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    omr = cw.ObservableMeasuredResult(
        setting=cw.InitObsSetting(
            init_state=cirq.Z(a) * cirq.Z(b),
            observable=cirq.Y(a) * cirq.Y(b),
        ),
        mean=0,
        variance=5 ** 2,
        repetitions=4,
        circuit_params={},
    )
    assert omr.stddev == 5
    assert omr.observable == cirq.Y(a) * cirq.Y(b)
    assert omr.init_state == cirq.Z(a) * cirq.Z(b)


@pytest.fixture()
def example_bsa() -> 'cw.BitstringAccumulator':
    """Test fixture to create an (empty) example BitstringAccumulator"""
    q0, q1 = cirq.LineQubit.range(2)
    setting = cw.InitObsSetting(
        init_state=cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1), observable=cirq.X(q0) * cirq.Y(q1)
    )
    meas_spec = _MeasurementSpec(
        max_setting=setting,
        circuit_params={
            'beta': 0.123,
            'gamma': 0.456,
        },
    )
    bsa = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=[
            setting,
            cw.InitObsSetting(init_state=setting.init_state, observable=cirq.X(q0)),
            cw.InitObsSetting(init_state=setting.init_state, observable=cirq.Y(q1)),
        ],
        qubit_to_index={q0: 0, q1: 1},
    )
    return bsa


def test_bitstring_accumulator(example_bsa):
    # test initialization
    assert example_bsa.bitstrings.shape == (0, 2)
    assert example_bsa.chunksizes.shape == (0,)
    assert example_bsa.timestamps.shape == (0,)

    # test consume_results
    bitstrings = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    example_bsa.consume_results(bitstrings)
    assert example_bsa.bitstrings.shape == (4, 2)
    assert example_bsa.chunksizes.shape == (1,)
    assert example_bsa.timestamps.shape == (1,)
    assert example_bsa.n_repetitions == 4

    # test results
    results = list(example_bsa.results)
    assert len(results) == 3
    for r in results:
        assert r.repetitions == 4

    # test records
    for r in example_bsa.records:
        assert isinstance(r, dict)
        assert 'repetitions' in r
        assert r['repetitions'] == 4


def test_bitstring_accumulator_strings(example_bsa):
    bitstrings = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    example_bsa.consume_results(bitstrings)

    q0, q1 = cirq.LineQubit.range(2)
    settings = cw.observables_to_settings(
        [
            cirq.X(q0),
            cirq.Y(q1),
            cirq.X(q0) * cirq.Y(q1),
        ],
        qubits=[q0, q1],
    )

    strings_should_be = [
        '+Z(0) * +Z(1) → X(0): 0.000 +- 0.577',
        '+Z(0) * +Z(1) → Y(1): 0.000 +- 0.577',
        '+Z(0) * +Z(1) → X(0)*Y(1): 0.000 +- 0.577',
    ]
    for setting, ssb in zip(settings, strings_should_be):
        assert example_bsa.summary_string(setting) == ssb, ssb

    assert (
        str(example_bsa)
        == """Accumulator +Z(0) * +Z(1) → X(0)*Y(1); 4 repetitions
  +Z(0) * +Z(1) → X(0)*Y(1): 0.000 +- 0.577
  +Z(0) * +Z(1) → X(0): 0.000 +- 0.577
  +Z(0) * +Z(1) → Y(1): 0.000 +- 0.577"""
    )


def test_bitstring_accumulator_equality():
    bitstrings = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    chunksizes = np.asarray([4])
    timestamps = np.asarray([datetime.datetime.now()])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    setting = cw.InitObsSetting(init_state=cirq.Z(a) * cirq.Z(b), observable=obs)
    meas_spec = _MeasurementSpec(setting, {})

    bsa = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=[setting],
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )

    bsa2 = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=[setting],
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )
    assert bsa == bsa2

    timestamps = np.asarray([datetime.datetime.now()])
    bsa3 = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=[setting],
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )
    assert bsa != bsa3


def test_bitstring_accumulator_stats():
    bitstrings = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    chunksizes = np.asarray([4])
    timestamps = np.asarray([datetime.datetime.now()])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    settings = list(
        cw.observables_to_settings(
            [cirq.Z(a) * cirq.Z(b) * 7, cirq.Z(a) * 5, cirq.Z(b) * 3], qubits=[a, b]
        )
    )
    meas_spec = _MeasurementSpec(settings[0], {})

    bsa = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=settings,
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )

    # There are three observables, each with mean 0 because
    # the four 2-bit strings have even numbers of a) ones in the
    # first position b) ones in the second position c) even parity
    # pairs.
    np.testing.assert_allclose([0, 0, 0], bsa.means())

    # Covariance: Sum[(x - xbar)(y - ybar)] / (N-1)
    # where xbar and ybar are 0, per above. Each individual observed
    # value is +-1, so (x-xbar)(y-bar) is +-1 (neglecting observable coefficients)
    # For off-diagonal elements, there are two +1 and two -1 terms for each entry
    # so the total contribution is zero, and the matrix is diagonal
    should_be = np.array(
        [
            [4 * 7 ** 2, 0, 0],
            [0, 4 * 5 ** 2, 0],
            [0, 0, 4 * 3 ** 2],
        ]
    )
    should_be = should_be / (4 - 1)  # covariance formula
    should_be = should_be / 4  # cov of the distribution of sample mean
    np.testing.assert_allclose(should_be, bsa.covariance())

    for setting, var in zip(settings, [4 * 7 ** 2, 4 * 5 ** 2, 4 * 3 ** 2]):
        np.testing.assert_allclose(0, bsa.mean(setting))
        np.testing.assert_allclose(var / 4 / (4 - 1), bsa.variance(setting))
        np.testing.assert_allclose(np.sqrt(var / 4 / (4 - 1)), bsa.stderr(setting))


def test_bitstring_accumulator_stats_2():
    bitstrings = np.array(
        [
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
        ]
    )
    chunksizes = np.asarray([4])
    timestamps = np.asarray([datetime.datetime.now()])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    settings = list(cw.observables_to_settings([cirq.Z(a) * 5, cirq.Z(b) * 3], qubits=[a, b]))
    meas_spec = _MeasurementSpec(settings[0], {})

    bsa = cw.BitstringAccumulator(
        meas_spec=meas_spec,
        simul_settings=settings,
        qubit_to_index=qubit_to_index,
        bitstrings=bitstrings,
        chunksizes=chunksizes,
        timestamps=timestamps,
    )

    # There are three observables, each with mean 0 because
    # the four 2-bit strings have even numbers of a) ones in the
    # first position b) ones in the second position.
    np.testing.assert_allclose([0, 0], bsa.means())

    # Covariance: Sum[(x - xbar)(y - ybar)] / (N-1)
    # where xbar and ybar are 0, per above. Each individual observed
    # value is +-1, so (x-xbar)(y-bar) is +-1 (neglecting observable coefficients)
    # In this case, the measurements are perfectly correlated.
    should_be = 4 * np.array(
        [
            [5 * 5, 5 * 3],
            [3 * 5, 3 * 3],
        ]
    )
    should_be = should_be / (4 - 1)  # covariance formula
    should_be = should_be / 4  # cov of the distribution of sample mean
    np.testing.assert_allclose(should_be, bsa.covariance())

    for setting, var in zip(settings, [4 * 5 ** 2, 4 * 3 ** 2]):
        np.testing.assert_allclose(0, bsa.mean(setting))
        np.testing.assert_allclose(var / 4 / (4 - 1), bsa.variance(setting))
        np.testing.assert_allclose(np.sqrt(var / 4 / (4 - 1)), bsa.stderr(setting))


def test_flatten_grouped_results():
    q0, q1 = cirq.LineQubit.range(2)
    settings = cw.observables_to_settings(
        [
            cirq.X(q0),
            cirq.Y(q0),
            cirq.Z(q0),
            cirq.Z(q0) * cirq.Z(q1),
        ],
        qubits=[q0, q1],
    )
    grouped_settings = cw.group_settings_greedy(settings)
    bsas = []
    for max_setting, simul_settings in grouped_settings.items():
        bsa = cw.BitstringAccumulator(
            meas_spec=_MeasurementSpec(max_setting, {}),
            simul_settings=simul_settings,
            qubit_to_index={q0: 0, q1: 1},
        )
        bsa.consume_results(
            np.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            )
        )
        bsas.append(bsa)

    results = cw.flatten_grouped_results(bsas)
    assert len(results) == 4
    for res in results:
        # We pass all 0's to each consume_results, so everything is 1 +- 0
        assert res.mean == 1
        assert res.variance == 0
        assert res.repetitions == 3