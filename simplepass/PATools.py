# Copyright 2020 - 2022, EMCORE Sp. z o.o.
# SPDX-License-Identifier: MIT

import enum
import itertools
import math
import psutil
import random
import scipy
import scipy.stats
import time

import chipwhisperer as cw
import numpy as np
import bokeh.plotting as bplt
import bokeh.palettes as bpal

from TracesFileProxy import TracesFileProxy_v2


def _in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if _in_notebook():
    from tqdm.notebook import tqdm, trange
    # Bokeh config
    bplt.output_notebook()
else:
    from tqdm.auto import tqdm, trange

palette = itertools.cycle(bpal.Category20_20[0::2] + bpal.Category20_20[1::2])

try:
    from math import comb
except ImportError:
    from math import factorial

    def comb(a, b):
        return factorial(a) // factorial(b) // factorial(a - b)


_hw_lookup_bits = 12
_hw_lookup_limit = 1 << _hw_lookup_bits
_hw_lookup_mask = _hw_lookup_limit - 1
HW = [bin(x).count('1') for x in range(_hw_lookup_limit)]


def hw(x):
    """
    Compute hamming weight of a given value.
    
    :param x: non-negative integer of arbitrary length or bytes object
    :returns: parameter's hamming weight
    :raises ValueError: raises an exception for unsupported input types
    """
    if isinstance(x, int):
        if x < _hw_lookup_limit:
            return HW[x]
        else:
            return HW[x & _hw_lookup_mask] + hw(x >> _hw_lookup_bits)
    elif isinstance(x, bytes):
        return sum(HW[b] for b in x)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")


class ScopeTarget:
    """
    The class provides the abstraction for basic ChipWhisperer scope and target management,
    is a utility class for gathering traces of device key-input-output functions,
    and provides a way to gather traces of device functions to files.
    
    :ivar scope: ChipWhisperer connected scope object
    :ivar target: ChipWhisperer connected target object
    :ivar base_baud: default baud rate of a ChipWhisperer-to-target connection
    :ivar base_clock: default clock of ChipWhisperer's target, defaults to 7370000
    :ivar _target_clock: desired target clock, set with :py_func:`~target_clock` property.
    :ivar key_bits: key buffer bit length, defaults to 128
    :ivar txi_bits: input buffer bit length, defaults to 128
    :ivar txo_bits: output buffer bit length, defaults to 128
    :ivar last_key: the last used key buffer value, initiated with an empty 'bytes' object
    """

    def __init__(self, scope_args, target_args):
        self.scope = cw.scope(*scope_args)
        self.target = cw.target(self.scope, *target_args)
        self.base_baud = self.target.baud
        self.base_clock = 7370000
        self._target_clock = self.base_clock
        self._target_metadata = {}

        self.key_bits = 128
        self.txi_bits = 128
        self.txo_bits = 128
        self.last_key = b''

        self.closed = False

    def close(self):
        self.target.close()
        self.scope.dis()
        self.closed = True

    @classmethod
    def _bits_to_bytes(cls, bits):
        """
        Convert number of bits to number of bytes required to store this number of bits.
        
        :param bits: number of bits
        :returns: number of bytes
        """
        return (bits + 7) // 8

    @property
    def clock_boost(self):
        """
        Target clock frequency multiplier. See :py_func:`~target_clock` for reference.
        """
        return self._target_clock / self.base_clock

    @property
    def target_clock(self):
        """
        Target clock frequency variable calculated as the base clock frequency multiplied by the boost.
        Note that the variable itself is independent from the actual clock frequency set on the target device
        and affects the device only when :py:func:`~set_clock` function is called.
        """
        return int(self._target_clock)

    @target_clock.setter
    def target_clock(self, value):
        """
        The setter of target_clock property, recalculates boost multiplier.
        Does not configure clock frequency on the target device.
        
        :param value: new target clock frequency
        """
        self._target_clock = value

    def set_clock(self):
        """
        Configure clock frequency and baud rate of the target device
        according to :py:attr:`~target_clock` property.
        """
        self.scope.clock.clkgen_freq = self._target_clock
        print(f"Scope clkgen clock rate = {self.scope.clock.clkgen_freq}")
        self.target.baud = int(self.base_baud * self.clock_boost)

    def reset_clock(self):
        """
        Reset clock frequency and baud rate of the target device to the base values.
        """
        self.scope.clock.clkgen_freq = self.base_clock
        print(f"Scope clkgen clock rate = {self.scope.clock.clkgen_freq}")
        self.target.baud = self.base_baud

    def reset_target(self):
        """
        Send reset signal to the target device.
        """
        self.scope.io.nrst = 'low'
        time.sleep(0.05)
        self.scope.io.nrst = 'high_z'
        time.sleep(0.05)
        self.last_key = b''

    def _scope_arm(self):
        return self.scope.arm()

    def _scope_capture(self):
        ret = self.scope.capture()
        if ret:
            print('Timeout happened during acquisition')
        trace = self.scope.get_last_trace()
        return trace

    def run_one(self, key=None, txi=None, repeat_malformed=True):
        """
        Send command 'p' to the target device with 'txi' parameter and gather the resulting power trace.
        Read the output buffer of the device after the execution.
        Additional 'key' value is set on the device with command 'k' if it has not been set before or if the 'key' value is different than the previous one.
        If 'key' or 'txi' parameter is None, the value of respective variable is chosen uniformly at random.
        
        :param key: key bytes of key_bits bit length, defaults to None
        :param txi: input bytes of txi_bits bit length, defaults to None
        :param repeat_malformed: indicates whether to try again if the device yields no readable output
        :returns: the tuple consisting of the power trace, sent key, sent input and device output
        """
        if key is None:
            key = random.getrandbits(self.key_bits).to_bytes(
                self._bits_to_bytes(self.key_bits), 'little')
        if key != self.last_key:
            try:
                self.target.simpleserial_write('k', key)
                self.target.simpleserial_wait_ack()
                self.last_key = key
            except:  # Unset last key, since it's in an unknown state
                self.last_key = b''
        if txi is None:
            txi = random.getrandbits(self.txi_bits).to_bytes(
                self._bits_to_bytes(self.txi_bits), 'little')
        self._scope_arm()
        self.target.simpleserial_write('p', txi)
        if self.txo_bits:
            txo = self.target.simpleserial_read(
                'r', self._bits_to_bytes(self.txo_bits))
            if not txo:
                print(f"Warning: malformed output for txi: {txi} key: {key}")
                if repeat_malformed:
                    return self.run_one(key, txi, repeat_malformed)
                txo = b''
            else:
                txo = bytes(txo)
        else:
            txo = None
        trace = self._scope_capture()
        return (trace, key, txi, txo)

    def gather_for_generator(self,
                             generator,
                             fname,
                             total_count,
                             batch_count=4096,
                             scope_samples=None,
                             track_progress=True,
                             dtype=np.float32):
        """
        Gather power traces of multiple device function executions and store them in a file.
        Function inputs and outputs are stored in a supplementary metadata file.
        Data are dumped into the files in batches of specified length.
        Refer to :py:obj:`~run_one` for the description of power trace gathering method.

        :param generator: device function input values generator, must generate (key, input) tuples
        :param fname: name of the files to store the traces and metadata (without file extensions)
        :param total_count: number of function executions to be performed and measured
        :param batch_count: size of the batch of function executions in which data is dumped into files
        :param scope_samples: maximum number of trace points per trace, if not provided taken as
                              :py:obj:`scope_samples`.
        :param track_progress: when True, uses TQDM to track trace gathering progress.
        :param dtype: data type used in the trace file.
        :returns: opened object of :py:class:`~TracesFileProxy.TracesFileProxy_v2` type used to store the data 
        """
        if scope_samples is None:
            scope_samples = self.scope_samples
        tracef = TracesFileProxy_v2.create(fname,
                                           total_count=total_count,
                                           max_length=scope_samples,
                                           dtype=dtype)
        tracef.global_meta.update(self.scope_metadata)
        tracef.global_meta.update(self.target_metadata)

        try:
            # Generator may be either a generator object, or a function; functions need to be called
            if callable(generator):
                generator = generator(total_count)
            # First, dump all metadata into the file, keeping 'captured': False
            for i, (key, txi) in enumerate(generator):
                # Break manually if generator doesn't track total_count
                if i > total_count: break
                tracef.trace_meta[i]['key'] = key
                tracef.trace_meta[i]['txi'] = txi
            tracef.flush()
            self.capture_uncaptured(tracef, batch_count, track_progress)
        except:
            tracef.close()
            raise
        return tracef

    def capture_uncaptured(self,
                           tracef: TracesFileProxy_v2,
                           batch_count=4096,
                           track_progress=True):
        """
        Capture all traces that are marked as `'captured': False` in `tracef` file.
        :param tracef: TracesFileProxy containing uncaptured traces
                       (traces with metadata but no (or obsolete) data)
        :param batch_count: every `batch_count` traces the trace file is flushed;
                            lower values require more frequent disk IO, but result in
                            more frequent "saves", higher values pose higher data
                            loss risk.
        :param track_progress: when True, uses TQDM to track trace gathering progress.
        """
        max_len = tracef.max_length
        self.reset_target()
        rgen = trange(tracef.total_count, desc='Capturing traces') \
               if track_progress else \
               range(tracef.total_count)
        for i in rgen:
            trace_meta = tracef.trace_meta[i]
            if not trace_meta['captured']:
                key = trace_meta['key']
                txi = trace_meta['txi']
                (trace, _, _, txo) = self.run_one(key, txi)
                trace = np.asarray(trace[:max_len], dtype=tracef.dtype)
                tlen = len(trace)
                tracef.traces[i, :tlen] = trace
                if tlen < max_len:
                    tracef.traces[i, tlen:] = np.zeros(max_len - tlen)
                trace_meta['txo'] = txo
                trace_meta['captured'] = True
                if i % batch_count == batch_count - 1:
                    tracef.flush()
        tracef.flush()

    @property
    def fname_infix(self):
        """
        Target/scope clock configuration identifier, used to name and identify trace files.
        """
        return f'{self._target_clock}-{self.scope.adc.offset}-{self.scope.adc.offset+self.scope.adc.samples}'
    
    @property
    def scope_metadata(self):
        return {
            'target_clock': self.target_clock,
            'window_begin': self.scope.adc.offset,
            'window_end': self.scope.adc.offset+self.scope.adc.samples,
            'scope': {
                'type': type(self.scope).__name__,
                'sn': self.scope.sn,
                'fw': self.scope.fw_version_str
            },
            'sample_rate': self.sample_rate,
        }
    
    @property
    def target_metadata(self):
        return self._target_metadata

    @property
    def scope_samples(self):
        return self.scope.adc.samples
    
    @property
    def sample_rate(self):
        return self.scope.clock.adc_rate


def bplotme(*traces, title=None):
    """
    Display multiple plots in one picture.
    
    :param traces: plots to be displayed
    :param title: title of the picture
    """
    p = bplt.figure(title=title)
    plot_legend = len(traces) > 1
    for i, (trace, color) in enumerate(zip(traces, palette)):
        lineargs = dict(x=range(len(trace)), y=trace, color=color)
        if plot_legend:
            lineargs['legend_label'] = f"Plot {i}"
        p.line(**lineargs)
    if plot_legend:
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
    bplt.show(p)


def bplotfft(*traces, T=None, S=None, N=None, title=None):
    p = bplt.figure(y_axis_type="log", title=title)
    plot_legend = len(traces) > 1
    if N is None:
        N = len(traces[0])
    if T is None and S is None:
        raise ValueError(
            "Either S or T must be provided (sampling rate or sample period)")
    if T is not None and S is not None:
        raise ValueError(
            "Only one of S or T may be provided (sampling rate or sample period)"
        )
    if T is None:
        T = 1 / S
    xt = scipy.fft.fftfreq(N, T)[:N // 2]
    for i, (trace, color) in enumerate(zip(traces, palette)):
        yt = scipy.fft.fft(trace)
        yv = 2.0 / N * np.abs(yt[:N // 2])
        lineargs = dict(x=xt, y=yv, color=color)
        if plot_legend:
            lineargs['legend_label'] = f"Plot {i}"
        p.line(**lineargs)
    if plot_legend:
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
    bplt.show(p)


def format_guess(real_key, key_guess, sk):
    """
    Print best 8 guesses and the rank of the correct value for a specified key buffer byte index.
    
    :param real_key: actual value of key buffer
    :param key_guess: collection of (byte value, additional data) tuples sorted descending by likeliness
    :param sk: key buffer byte index for which the statistics are to be printed
    """
    rk = real_key[sk]
    rank = [j for j, b in enumerate(key_guess) if rk == b[0]][0]
    print(f'Best guesses for byte {sk} (correct value {rk:02x}, rank {rank}):')
    for b, r in key_guess[:8]:
        print(f'{b:02x} ({r})')


def format_guesses(real_key, key_guess, weight_progression=False, weight_key=None):
    """
    Print summary for the full key guess with the most likely byte values.
    The guess is compared against the actual key;
    the summary includes bitwise difference and the rank of the correct value per each byte.
    
    :param real_key: actual value of key buffer
    :param key_guess: collections of (byte value,  additional data) tuples sorted descending by likeliness,
                      iterable with key byte index
    :param weight_progression: when true, print weights for every 2^ith guess.
    :param weight_key: when weight_progression, defines how to extract kg weight from additional data. Defaults to full additional_data
    """
    # yapf: disable
    print('Real key:  ', ' '.join('{:02x}'.format(rk) for rk in real_key))
    print('Best guess:', ' '.join('{:02x}'.format(bg[0][0]) for bg in key_guess))
    print('Diff:      ', ' '.join('{:02x}'.format(rk ^ bg[0][0])
                                  for (rk, bg) in zip(real_key, key_guess)))
    print('Byte rank: ', ' '.join('{:02x}'.format([j for j, b in enumerate(bg)
                                                   if rk == b[0]][0])
                                  for (rk, bg) in zip(real_key, key_guess)))
    # yapf: enable
    if weight_progression:
        if weight_key is None:
            weight_key = lambda x: x
        print("Guess weight progression:")
        table = {}
        maxi = 0
        maxc = 0
        for x, kg in enumerate(key_guess):
            i = 0
            j = 1  # = 2^0
            while j <= len(kg):
                table[j-1, x] = format(weight_key(kg[j-1][1]), '.4g')
                maxc = max(maxc, len(table[j-1, x]))
                j *= 2
                i += 1
            maxi = max(maxi, i)
        maxjlen = len(format((2**(maxi-1)-1)))
        j = 1
        kgrange = range(len(key_guess))
        for i in range(maxi):
            cellvs = (
                (table[j-1, x] if (j-1, x) in table else '') for x in kgrange
            )
            cells = (
                format(cell, f'{maxc}s') for cell in cellvs
            )
            print(format(j-1, f'{maxjlen}d'), *cells)
            j *= 2


def format_guess_hw(real_key, hw_guess, sk):
    rk = real_key[sk]
    print(
        f'Guesses for byte {sk} (correct value {rk:02x}, correct hw {hw(rk)}):'
    )
    for h, r in hw_guess:
        print(f'{h} ({r})')


def format_guesses_hw(real_key, key_guess):
    # yapf: disable
    print('Real key:  ', ' '.join('{:02x}'.format(rk) for rk in real_key))
    print('Base hws:  ', ' '.join('{:2}'.format(hw(rk)) for rk in real_key))
    print('Best guess:', ' '.join('{:2}'.format(bg[0][0]) for bg in key_guess))
    print('HW rank:   ', ' '.join('{:2}'.format([j for j, b in enumerate(bg)
                                                 if hw(rk) == b[0]][0])
                                  for (rk, bg) in zip(real_key, key_guess)))
    # yapf: enable


# Correlation Power Analysis functions


def corr_traces(metric, traces, mean_metric=None, mean_trace=None):
    """
    Compute correlation coefficient for each trace sample with the metric.
    
    :param metric: metric of size [:n]
    :param traces: traces of dimension [:n,:m]
    :param mean_metric: mean of the metric; if None, metric numpy mean is calculated
    :param mean_trace: mean of the traces; if None, traces numpy average is calculated
    :returns: row corr[:m] where corr[i] = corrcoef(metric[:n], traces[:n])
    """
    n = len(metric)
    m = len(traces[0])
    if mean_metric is None:
        mean_metric = np.mean(metric, dtype=np.float64)
    if mean_trace is None:
        mean_trace = np.average(traces, axis=0, dtype=np.float64)

    covht = np.zeros(m, dtype=np.float64)
    varh = 0
    vart = np.zeros(m, dtype=np.float64)

    for d in range(n):
        hdiff = metric[d] - mean_metric
        tdiff = traces[d] - mean_trace
        covht += hdiff * tdiff
        varh += hdiff**2
        vart += tdiff**2

    corr = covht / np.sqrt(varh * vart)
    np.nan_to_num(corr, copy=False)
    return corr


def corr_traces_pre(metric, d_traces, nvar_trace, mean_metric=None):
    """
    Compute correlation coefficient for each trace sample with the metric.
    Contrary to :py:func:`~corr_traces`, the function uses precomputed values
    of point-wise differences to mean for each of n traces, and traces point-wise n*variance.
    
    :param metric: metric of size [:n]
    :param d_traces: traces point-wise differences to mean trace of dimension [:n,:m]
    :param nvar_trace: point-wise n*variance of n traces (sum of squares of differences to mean, i.e. n*Var(traces))
    :param mean_metric: mean of the metric; if None, metric numpy mean is calculated
    :returns: row corr[:m] where corr[i] = corrcoef(metric[:n], traces[:n])
    """
    dtype = d_traces.dtype
    if mean_metric is None:
        mean_metric = np.mean(metric, dtype=dtype)
    d_metrics = np.asarray(metric, dtype=dtype) - mean_metric
    nvar_metric = 0
    for diff in d_metrics:
        nvar_metric += diff * diff

    covht = np.matmul(d_traces.T, d_metrics)
    corr = covht / np.sqrt(nvar_metric * nvar_trace)
    np.nan_to_num(corr, copy=False)
    return corr


def corr_traces_many_pre(d_metrics, nvar_metric, d_traces, nvar_trace):
    """
    Compute correlation coefficient for each trace sample with the metric.
    Contrary to :py:func:`~corr_traces_pre`, the function computes results for many metrics,
    resulting in a matrix of size [:m,:o], where:
     * n is the number of traces
     * l is the number of trace ponts (trace length)
     * m is the number of metrics
    The function uses metrics in a precomputed diff form, which can be obtained the same
    way as precomputed traces, e.g. using :py:func:`~precompute_difftraces`.
    
    :param d_metric:   metric differences array of size [:n,:m]
    :param nvar_metri: metric n*variance array of size [:m]
    :param d_traces:   traces point-wise differences to mean trace of dimension [:n,:l]
    :param nvar_trace: point-wise n*variance of n traces (sum of squares of differences to mean, i.e. n*Var(traces)) array of size [:l]
    :returns: matrix corr[:m,:l] where corr[i,j] = corrcoef(metric[:n,j], traces[:n,i])
    """
    out = np.matmul(d_metrics.T, d_traces) \
        / np.sqrt(np.outer(nvar_metric, nvar_trace))
    np.nan_to_num(out, copy=False)
    return out


def precompute_difftraces(traces):
    """
    Compute mean trace, point-wise differences to mean for each of n traces,
    and point-wise n*variance of n traces (sum of squares of differences to mean, i.e. n*Var(traces)).
    
    :param traces: traces of dimension [:n,:m]
    :returns: (mean trace of size [:m], differences to mean of dimension [:n,:m], n*variance of size [:m]) tuple
    """
    dtype = traces.dtype
    trace_cnt, trace_len = traces.shape
    meantrace = np.average(traces, axis=0)
    difftraces = np.empty((trace_cnt, trace_len), dtype=dtype)
    nvar_trace = np.zeros(trace_len, dtype=dtype)
    for i, trace in enumerate(traces):
        difftraces[i] = trace - meantrace
        nvar_trace += difftraces[i]**2
    return meantrace, difftraces, nvar_trace


class CPAModel(enum.Enum):
    ABS = enum.auto()
    POS = enum.auto()
    NEG = enum.auto()
    SIG = enum.auto()


class CPACollapseMethod(enum.Enum):
    MAX = enum.auto()
    MIN = enum.auto()
    AVG = enum.auto()


def single_cpa(metric_func,
               difftraces: np.ndarray,
               nvar_trace: np.ndarray,
               inputs,
               guess_range,
               *,
               plot_ris=False,
               model=CPAModel.ABS,
               best_reduce=CPACollapseMethod.MAX,
               corr_signs=None):
    """
    Compute correlation power analysis ranking for a given metric function, set of traces with corresponding input, and guess range
    
    :param metric_func: leakage function of form (guess from guess_range, input from inputs) -> value
    :param difftraces: traces point-wise differences to mean trace of dimension [:n,:m]
    :param nvar_trace: point-wise n*variance of n traces (sum of squares of differences to mean, i.e. n*Var(traces))
    :param inputs: input metadata used for each trace (can be e.g. a tuple of input and output)
    :param guess_range: generator for guess parameter of the metric function
    :param plot_ris: plot best correlation coefficients per guess
    :param model: correlation model; abs takes the best absolute correlation,
                  neg takes the best negative and pos the best positive
    :param best_reduce: function collapsing correlation values from each trace point to a single
                        weight; few common examples:
                         * MAX - default collapsing function, best used when only a few trace points correlate
                         * MIN - best used when all trace points have strong correlation
                         * AVG - best used when most but not all trace point have strong correlation
    :param corr_signs: only for model == SIG; an array or list of correlation signs (1 or -1)
    :returns: collection of (guess value, (correlation coefficient, trace point)) tuples
              sorted descending by highest correlation, where trace point is the point
              with the highest correlation
    """
    if model is CPAModel.SIG:
        ext = lambda rcorr, i, j: rcorr[i, j] * corr_signs[j]
    else:
        mod = {
            CPAModel.ABS: abs,
            CPAModel.POS: lambda x: x,
            CPAModel.NEG: lambda x: -x
        }[model]
        ext = lambda rcorr, i, j: mod(rcorr[i, j])
    reduce = {
        CPACollapseMethod.MAX: max,
        CPACollapseMethod.MIN: min,
        CPACollapseMethod.AVG: lambda x: (sum(e for e, i in x) / trace_len, 0),
    }[best_reduce]
    dtype = difftraces.dtype
    # Convert guesses to a list with known length; this list is later used to hold results
    guesses = list(guess_range)
    trace_cnt, trace_len = difftraces.shape
    guess_cnt = len(guesses)
    # Here we estimate the available and required memory for a given number of guesses;
    # this allows us to split into maximal chunks
    estimate_margin = 1.1
    estimated_mem_for_guess = int(estimate_margin * dtype.itemsize * (
        2 * trace_cnt +  # metrics, diffmetrics
        2 +  # nvar_metric, meanmetric
        3 * trace_len  # corr_traces & temporary
    ))
    import gc
    gc.collect()  # Force GC before checking for available virtual memory
    mem_available = psutil.virtual_memory().available
    max_guesses_per_iter = mem_available // estimated_mem_for_guess
    estimated_total_mem = guess_cnt * estimated_mem_for_guess

    def _process_guesses(guesses, start, stop):
        guess_slice = guesses[start:stop]
        guess_cnt = len(guess_slice)
        metrics = np.empty((trace_cnt, guess_cnt), dtype=dtype)
        # Iterate over guesses and apply hypothesis function for each guess
        for i, input_ in enumerate(tqdm(inputs)):
            for k, guess in enumerate(guess_slice):
                metrics[i, k] = metric_func(guess, input_)
        _, diffmetrics, nvar_metric = precompute_difftraces(metrics)
        rcorr = corr_traces_many_pre(diffmetrics, nvar_metric, difftraces,
                                     nvar_trace)
        for i in range(guess_cnt):
            guesses[i + start] = (guesses[i + start],
                                  reduce((ext(rcorr, i, j), j)
                                         for j in range(trace_len)))

    if len(guesses) < max_guesses_per_iter:
        _process_guesses(guesses, 0, guess_cnt)
    else:
        print(
            f"CPA: total memory required {estimated_total_mem/(1<<30):.2f} GiB, splitting into {int(math.ceil(guess_cnt / max_guesses_per_iter))} chunks"
        )
        for start in trange(0, guess_cnt, max_guesses_per_iter):
            _process_guesses(guesses, start, start + max_guesses_per_iter)

    # Plot before sort - this way the order of guesses is preserved on plot
    if plot_ris:
        bplotme([r[1][0] for r in guesses], title="Best correlations")
    guesses.sort(key=lambda x: -x[1][0])
    return guesses


def cpa_guess_key_pre(leak_fun,
                      difftraces,
                      nvar_trace,
                      txis,
                      real_key=None,
                      verbose=True,
                      model=CPAModel.ABS):
    """
    Compute correlation power analysis ranking for each key byte, using power traces and known inputs.
    Contrary to :py:func:`~cpa_guess_key`, the function uses precomputed values
    of point-wise differences to mean for each of n traces and traces point-wise n*variance.
    
    :param leak_fun: leakage function of form (key_byte_idx:int, guess:int, input:bytes) -> value
    :param difftraces: traces point-wise differences to mean trace of dimension [:n,:m]
    :param nvar_trace: point-wise n*variance of n traces (sum of squares of differences to mean, i.e. n*Var(traces))
    :param txis: inputs used for each trace
    :param real_key: actual secret key used in function (used solely to present the quality of results)
    :param model: correlation model; see :py:func:`single_cpa`
    :returns: collections of (byte value, correlation coefficient) tuples sorted descending by highest correlation,
              iterable with key byte index
    """
    best_guesses = []
    for sk in range(16):
        guesses_sorted = single_cpa(lambda key, txi: leak_fun(sk, key, txi),
                                    difftraces,
                                    nvar_trace,
                                    txis,
                                    range(256),
                                    model=model)
        if verbose and real_key:
            format_guess(real_key, guesses_sorted, sk)
        best_guesses.append(guesses_sorted)
    return best_guesses


def cpa_guess_key(leak_fun,
                  traces,
                  txis,
                  real_key=None,
                  verbose=True,
                  model=CPAModel.ABS):
    """
    Compute correlation power analysis ranking for each key byte, using power traces and known inputs.
    
    :param leak_fun: leakage function of form (key_byte_idx:int, guess:int, input:bytes) -> value
    :param traces: traces
    :param txis: inputs used for each trace
    :param real_key: actual secret key used in function (used solely to present quality of the results)
    :param model: correlation model; see :py:func:`single_cpa`
    :returns: collections of (byte value,  correlation coefficient) tuples sorted descending by highest correlation,
              iterable with key byte index
    """
    _, difftraces, nvar_trace = precompute_difftraces(traces)
    return cpa_guess_key_pre(leak_fun,
                             difftraces,
                             nvar_trace,
                             txis,
                             real_key=real_key,
                             verbose=verbose,
                             model=model)


# Differential Power Analysis


def single_dpa(traces,
               textins,
               guess_gen,
               sel_func,
               nbits=8,
               reductor=lambda x: np.sqrt(np.sum(np.square(x), axis=0))):
    # notable reductors:
    # * Take max deviation for each bit:
    #     lambda x: np.max(np.abs(x), axis=1)
    # * Take abs vector length at each point:
    #     lambda x: np.sqrt(np.sum(np.square(x), axis=0))
    ris = {}
    for guess in guess_gen:
        buckets = [([np.zeros(traces.shape[1], dtype=np.float64), 0],  # bucket 0
                    [np.zeros(traces.shape[1], dtype=np.float64), 0])  # bucket 1
                   for _ in range(nbits)]
        for trace, txi in zip(traces, textins):
            sel = sel_func(guess, txi)
            for i, b in enumerate(sel):
                buckets[i][b][0] += trace
                buckets[i][b][1] += 1
        rs = np.ndarray(shape=(nbits, traces.shape[1]), dtype=np.float64)
        for i, ((sum0, cnt0), (sum1, cnt1)) in enumerate(buckets):
            if not cnt0 or not cnt1:
                print("WARNING: empty bucket at bit", i, "guess", guess)
                rs[i] = np.zeros(traces.shape[1], dtype=np.float32)
            else:
                rs[i] = sum0 / cnt0 - sum1 / cnt1
        rs = reductor(rs)
        ris[guess] = max(((r, i) for i, r in enumerate(rs)),
                         key=lambda x: x[0])

    bplotme([r[0] for r in ris.values()], title="Best correlations")

    ris_sorted = list(ris.items())
    ris_sorted.sort(key=lambda x: -x[1][0])
    return ris_sorted


def gen_layer(layer, iter_lens):
    if len(iter_lens) > 1:
        curr_it = min(layer, iter_lens[0] - 1)
        while curr_it >= 0:
            sub_layer = layer - curr_it
            if sub_layer >= 0:
                for sub in gen_layer(sub_layer, iter_lens[1:]):
                    yield (curr_it, *sub)
            curr_it -= 1
    else:
        if layer < iter_lens[0]:
            yield (layer, )


def layer_iter(*iterables):
    iter_lens = [len(iterable) for iterable in iterables]
    total_layers = sum(iter_len - 1 for iter_len in iter_lens) + 1
    for layer in range(total_layers):
        # Iterate over all indices s.t. sum of indices = layer number
        for indices in gen_layer(layer, iter_lens):
            yield tuple(it[idx] for it, idx in zip(iterables, indices))


# Template Attack functions
#from TemplateAttack import TemplateAttack
