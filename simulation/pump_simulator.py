"""Approximate triplex-pump signal simulator and 14-feature extractor.

This is a Python approximation, not the MATLAB Simscape physical model.
Sampling follows the paper's 2 s measurement at 1 kHz; windows can be
scheduled every 30 s by the experiment runner.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import kurtosis, skew

FEATURES = ["fPeak","pLow","pMid","pHigh","pKurtosis","qMean","qVar",
            "qSkewness","qKurtosis","qPeak2Peak","qCrest","qRMS","qMAD","qCSRange"]

def _signal_stats(x):
    x=np.asarray(x,dtype=float)
    return {
        "mean": float(np.mean(x)),
        "var": float(np.var(x)),
        "skew": float(skew(x, bias=False)) if np.std(x)>1e-12 else 0.0,
        "kurt": float(kurtosis(x, fisher=False, bias=False)) if np.std(x)>1e-12 else 3.0,
        "p2p": float(np.ptp(x)),
        "crest": float(np.max(np.abs(x))/(np.sqrt(np.mean(x*x))+1e-12)),
        "rms": float(np.sqrt(np.mean(x*x))),
        "mad": float(np.median(np.abs(x-np.median(x)))),
        "csrange": float(np.quantile(x,.95)-np.quantile(x,.05)),
    }

def simulate_measurement(failure="normal", severity=0.0, seed=None, fs=1000, duration=2.0):
    """Generate one synthetic flow-pressure measurement with progressive leak severity [0,1]."""
    rng=np.random.default_rng(seed)
    n=int(fs*duration); t=np.arange(n)/fs
    severity=float(np.clip(severity,0,1))
    # Flow waveform: pump pulsation plus harmonics and stochastic noise.
    flow_mean=1.0
    pressure_level=1.0
    if failure=="leak":
        flow_mean -= .22*severity
        pressure_level -= .12*severity
    elif failure=="blocked_inlet":
        flow_mean -= .30*severity
    elif failure=="bearing_wear":
        pass
    pulse=(.12+.04*severity)*np.sin(2*np.pi*25*t)
    harmonics=.035*np.sin(2*np.pi*50*t+.2)+.015*np.sin(2*np.pi*75*t)
    noise=rng.normal(0,.025+.025*severity,n)
    q=flow_mean+pulse+harmonics+noise
    if failure=="bearing_wear":
        q += .05*severity*np.sin(2*np.pi*180*t)
    p=pressure_level+.08*np.sin(2*np.pi*25*t+.4)+rng.normal(0,.018+.02*severity,n)
    # Frequency-domain pressure descriptors.
    spec=np.abs(np.fft.rfft(p-np.mean(p)))
    freqs=np.fft.rfftfreq(n,1/fs)
    band=lambda lo,hi: float(np.max(spec[(freqs>=lo)&(freqs<hi)])) if np.any((freqs>=lo)&(freqs<hi)) else 0.0
    ps=_signal_stats(p); qs=_signal_stats(q)
    fpeak=float(freqs[int(np.argmax(spec[1:])+1)]) if len(spec)>1 else 0.0
    return {
      "fPeak":fpeak, "pLow":band(1,100), "pMid":band(100,300),
      "pHigh":band(300,500), "pKurtosis":ps["kurt"],
      "qMean":qs["mean"], "qVar":qs["var"], "qSkewness":qs["skew"],
      "qKurtosis":qs["kurt"], "qPeak2Peak":qs["p2p"], "qCrest":qs["crest"],
      "qRMS":qs["rms"], "qMAD":qs["mad"], "qCSRange":qs["csrange"]
    }

def generate_experiment(n_normal=240, n_leak=240, seed=42):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n_normal):
        rows.append({**simulate_measurement("normal",0, int(rng.integers(1,2**31-1))),"failure":0,"scenario":"normal","severity":0.0,"elapsed_seconds":i*30})
    for i in range(n_leak):
        sev=min(1.0,(i+1)/n_leak)
        rows.append({**simulate_measurement("leak",sev,int(rng.integers(1,2**31-1))),"failure":1,"scenario":"leak","severity":sev,"elapsed_seconds":(i+1)*30})
    return rows
