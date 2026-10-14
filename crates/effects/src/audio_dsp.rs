//! Audio DSP and frequency analysis foundation.
//!
//! Provides a calibrated Fast Fourier Transform (FFT) analysis engine capable of:
//! - Windowing (Hann) with exact energy normalization (Parseval-conserving).
//! - Single-FFT evaluation across multiple configurable frequency bands.
//! - Accurate low-frequency resolution (e.g. 10–125 Hz) given sufficient sample duration.
//! - Reusable scratch buffers (`FrequencyAnalyzer`) to eliminate per-frame allocations during playback and rendering.

use std::f64::consts::PI;

/// In-place radix-2 Cooley-Tukey decimation-in-time Fast Fourier Transform.
/// `re` and `im` must have identical lengths, which must be a power of two.
pub fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    if n < 2 || n != im.len() || !n.is_power_of_two() {
        return;
    }
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let (ws, wc) = ang.sin_cos();
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, bb) = (start + k, start + k + len / 2);
                let tr = re[bb] * cr - im[bb] * ci;
                let ti = re[bb] * ci + im[bb] * cr;
                re[bb] = re[a] - tr;
                im[bb] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wc - ci * ws;
                ci = cr * ws + ci * wc;
                cr = ncr;
            }
        }
        len <<= 1;
    }
}

/// Channel selection mode when analyzing stereo interleaved audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChannelMode {
    #[default]
    Mono,
    Left,
    Right,
}

/// A frequency band range specification [min_hz, max_hz].
#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyBand {
    pub name: String,
    pub min_hz: f64,
    pub max_hz: f64,
}

impl FrequencyBand {
    pub fn new(name: impl Into<String>, min_hz: f64, max_hz: f64) -> Self {
        Self { name: name.into(), min_hz, max_hz }
    }
}

/// Analyzed amplitude outputs for a single frequency band.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BandOutput {
    /// Peak-equivalent amplitude of a sine wave in this band, normalized such that
    /// a full-scale 0 dBFS pure sine wave yields 1.0. Range: [0.0, ..).
    pub peak_amplitude: f64,
    /// Root-mean-square amplitude (energy / power). For a pure sine wave: `peak / sqrt(2)`.
    pub rms_amplitude: f64,
    /// Value scaled for slider parameters (0.0..100.0, where 100.0 = 0 dBFS peak).
    pub slider_value: f64,
    /// Mean square spectral power in the band.
    pub power: f64,
}

/// Single-FFT spectral representation of an audio window.
#[derive(Clone, Debug, Default)]
pub struct SpectrumAnalysis {
    /// The audio sample rate (Hz).
    pub rate: u32,
    /// Number of real audio samples before zero-padding.
    pub sample_count: usize,
    /// FFT size (power of two, >= sample_count).
    pub fft_size: usize,
    /// Frequency spacing between FFT bins (Hz).
    pub hz_per_bin: f64,
    /// One-sided normalized power spectrum for bins 0..=fft_size/2.
    /// The sum of power across all bins corresponds to time-domain mean-square energy
    /// (0.5 * A^2 for a full-scale sine of peak A).
    pub power_spectrum: Vec<f64>,
}

impl SpectrumAnalysis {
    /// Compute spectral analysis from mono f64 audio samples.
    pub fn from_mono(samples: &[f64], rate: u32) -> Self {
        let mut analyzer = FrequencyAnalyzer::new();
        analyzer.analyze_mono(samples, rate)
    }

    /// Compute spectral analysis from stereo interleaved f32 PCM samples ([L, R, L, R, ...]).
    pub fn from_stereo_interleaved(samples: &[f32], rate: u32, channel: ChannelMode) -> Self {
        let mut analyzer = FrequencyAnalyzer::new();
        analyzer.analyze_stereo(samples, rate, channel)
    }

    /// Analyze a frequency band [min_hz, max_hz].
    ///
    /// Safely handles:
    /// - Inverted ranges (min > max).
    /// - Negative frequencies (clamped to 0.0).
    /// - Frequencies above the Nyquist limit (clamped to rate / 2).
    /// - DC offset exclusion (unless min_hz == 0.0).
    /// - Silence or empty input (returns zero amplitude).
    pub fn band(&self, min_hz: f64, max_hz: f64) -> BandOutput {
        if self.power_spectrum.is_empty() || self.hz_per_bin <= 0.0 || !min_hz.is_finite() || !max_hz.is_finite() {
            return BandOutput::default();
        }
        let (mut lo, mut hi) = if min_hz <= max_hz { (min_hz, max_hz) } else { (max_hz, min_hz) };
        lo = lo.max(0.0);
        let nyquist = self.rate as f64 * 0.5;
        if lo >= nyquist {
            return BandOutput::default();
        }
        hi = hi.clamp(lo, nyquist);

        let max_bin = self.power_spectrum.len().saturating_sub(1);
        // Exclude DC (bin 0) for AC audio bands unless 0.0 Hz was explicitly requested as the lower bound.
        let k_start = if lo <= 0.0 {
            0
        } else {
            ((lo / self.hz_per_bin).floor() as usize).max(1).min(max_bin)
        };
        let k_end = ((hi / self.hz_per_bin).ceil() as usize).clamp(k_start, max_bin);

        let mut power = 0.0f64;
        for k in k_start..=k_end {
            if let Some(&p) = self.power_spectrum.get(k) {
                power += p;
            }
        }

        let rms = power.max(0.0).sqrt();
        let peak = (2.0 * power).max(0.0).sqrt();
        let slider = (peak * 100.0).clamp(0.0, 1000.0);

        BandOutput {
            peak_amplitude: peak,
            rms_amplitude: rms,
            slider_value: slider,
            power,
        }
    }

    /// Analyze multiple frequency bands from this single precomputed FFT.
    pub fn bands(&self, bands: &[FrequencyBand]) -> Vec<BandOutput> {
        bands.iter().map(|b| self.band(b.min_hz, b.max_hz)).collect()
    }
}

/// Reusable scratch buffers for frame-by-frame analysis without repeated heap allocations.
#[derive(Default)]
pub struct FrequencyAnalyzer {
    re: Vec<f64>,
    im: Vec<f64>,
    power_spectrum: Vec<f64>,
    mono_buf: Vec<f64>,
}

impl FrequencyAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze mono audio samples, reusing internal buffers.
    pub fn analyze_mono(&mut self, samples: &[f64], rate: u32) -> SpectrumAnalysis {
        let m = samples.len();
        if m < 2 || rate == 0 {
            return SpectrumAnalysis {
                rate,
                sample_count: m,
                fft_size: 0,
                hz_per_bin: 0.0,
                power_spectrum: Vec::new(),
            };
        }
        let n = m.next_power_of_two().max(64);
        if self.re.len() < n {
            self.re.resize(n, 0.0);
            self.im.resize(n, 0.0);
        } else {
            self.re[..n].fill(0.0);
            self.im[..n].fill(0.0);
        }

        let denom = (m - 1) as f64;
        let mut window_energy = 0.0f64;
        for (i, &s) in samples.iter().enumerate() {
            let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / denom).cos();
            window_energy += w * w;
            if let Some(r) = self.re.get_mut(i) {
                *r = s * w;
            }
        }

        fft(&mut self.re[..n], &mut self.im[..n]);

        let hz_per_bin = rate as f64 / n as f64;
        let num_bins = n / 2 + 1;
        let norm = if window_energy > 0.0 && n > 0 {
            1.0 / (n as f64 * window_energy)
        } else {
            0.0
        };

        self.power_spectrum.clear();
        self.power_spectrum.reserve(num_bins);
        for k in 0..num_bins {
            let r = self.re.get(k).copied().unwrap_or(0.0);
            let i_val = self.im.get(k).copied().unwrap_or(0.0);
            let mag_sq = r * r + i_val * i_val;
            let factor = if k == 0 || k * 2 == n { 1.0 } else { 2.0 };
            self.power_spectrum.push(mag_sq * factor * norm);
        }

        SpectrumAnalysis {
            rate,
            sample_count: m,
            fft_size: n,
            hz_per_bin,
            power_spectrum: self.power_spectrum.clone(),
        }
    }

    /// Analyze stereo interleaved audio samples, reusing internal buffers.
    pub fn analyze_stereo(&mut self, samples: &[f32], rate: u32, channel: ChannelMode) -> SpectrumAnalysis {
        let mut mono = std::mem::take(&mut self.mono_buf);
        mono.clear();
        mono.reserve(samples.len() / 2);
        for c in samples.chunks_exact(2) {
            let l = c[0] as f64;
            let r = c[1] as f64;
            match channel {
                ChannelMode::Mono => mono.push((l + r) * 0.5),
                ChannelMode::Left => mono.push(l),
                ChannelMode::Right => mono.push(r),
            }
        }
        let res = self.analyze_mono(&mono, rate);
        self.mono_buf = mono;
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sine(freq_hz: f64, amp: f64, duration_s: f64, rate: u32) -> Vec<f64> {
        let count = (duration_s * rate as f64) as usize;
        (0..count)
            .map(|i| {
                let t = i as f64 / rate as f64;
                amp * (2.0 * PI * freq_hz * t).sin()
            })
            .collect()
    }

    #[test]
    fn bin_centered_and_off_bin_sine_calibration() {
        const RATE: u32 = 48_000;
        // 200 ms at 48 kHz = 9600 samples, padded to n = 16384 -> hz_per_bin = 48000 / 16384 ≈ 2.9296875 Hz.
        // Bin 34 is 34 * 2.9296875 = 99.609375 Hz (near 100 Hz).
        let on_bin_freq = 34.0 * (RATE as f64 / 16384.0);
        let off_bin_freq = 100.0; // Between bin 34 (99.6 Hz) and bin 35 (102.5 Hz).

        let on_bin = generate_sine(on_bin_freq, 1.0, 0.2, RATE);
        let off_bin = generate_sine(off_bin_freq, 1.0, 0.2, RATE);

        let spec_on = SpectrumAnalysis::from_mono(&on_bin, RATE);
        let spec_off = SpectrumAnalysis::from_mono(&off_bin, RATE);

        let out_on = spec_on.band(30.0, 250.0);
        let out_off = spec_off.band(30.0, 250.0);

        // Peak amplitude of 1.0 should yield slider value ~ 100.0 (tolerance 2%).
        assert!(
            (out_on.slider_value - 100.0).abs() < 2.0,
            "on-bin amplitude should be ~100.0, got {}",
            out_on.slider_value
        );
        assert!(
            (out_off.slider_value - 100.0).abs() < 2.0,
            "off-bin amplitude should be ~100.0, got {}",
            out_off.slider_value
        );
        assert!(
            (out_on.slider_value - out_off.slider_value).abs() < 1.0,
            "scalloping loss should be minimal across bins: on={}, off={}",
            out_on.slider_value,
            out_off.slider_value
        );
    }

    #[test]
    fn linearity_across_input_amplitudes() {
        const RATE: u32 = 48_000;
        for &amp in &[1.0, 0.75, 0.5, 0.25, 0.1, 0.01] {
            let samples = generate_sine(200.0, amp, 0.2, RATE);
            let spec = SpectrumAnalysis::from_mono(&samples, RATE);
            let out = spec.band(100.0, 300.0);
            let expected = amp * 100.0;
            assert!(
                (out.slider_value - expected).abs() < 1.5,
                "amplitude {amp} expected slider ~{expected}, got {}",
                out.slider_value
            );
            assert!(
                (out.peak_amplitude - amp).abs() < 0.015,
                "amplitude {amp} expected peak ~{amp}, got {}",
                out.peak_amplitude
            );
        }
    }

    #[test]
    fn in_band_vs_out_of_band_rejection() {
        const RATE: u32 = 48_000;
        // 1 kHz sine wave
        let samples = generate_sine(1000.0, 1.0, 0.2, RATE);
        let spec = SpectrumAnalysis::from_mono(&samples, RATE);

        let in_band = spec.band(800.0, 1200.0);
        let out_band_low = spec.band(30.0, 250.0);
        let out_band_high = spec.band(3000.0, 8000.0);

        assert!((in_band.slider_value - 100.0).abs() < 2.0);
        assert!(out_band_low.slider_value < 0.2, "out of band low should be rejected, got {}", out_band_low.slider_value);
        assert!(out_band_high.slider_value < 0.2, "out of band high should be rejected, got {}", out_band_high.slider_value);
    }

    #[test]
    fn multiple_simultaneous_tones_in_different_bands() {
        const RATE: u32 = 48_000;
        let s_bass = generate_sine(60.0, 0.6, 0.2, RATE);
        let s_mid = generate_sine(800.0, 0.4, 0.2, RATE);
        let s_high = generate_sine(4000.0, 0.9, 0.2, RATE);

        let mixed: Vec<f64> = s_bass.iter().zip(&s_mid).zip(&s_high).map(|((b, m), h)| b + m + h).collect();
        let spec = SpectrumAnalysis::from_mono(&mixed, RATE);

        let bands = vec![
            FrequencyBand::new("Bass", 30.0, 150.0),
            FrequencyBand::new("Mid", 500.0, 1500.0),
            FrequencyBand::new("High", 3000.0, 6000.0),
            FrequencyBand::new("Unused", 10000.0, 15000.0),
        ];

        let results = spec.bands(&bands);
        assert!((results[0].slider_value - 60.0).abs() < 2.0, "bass expected ~60.0, got {}", results[0].slider_value);
        assert!((results[1].slider_value - 40.0).abs() < 2.0, "mid expected ~40.0, got {}", results[1].slider_value);
        assert!((results[2].slider_value - 90.0).abs() < 2.0, "high expected ~90.0, got {}", results[2].slider_value);
        assert!(results[3].slider_value < 0.2, "unused band should be near 0, got {}", results[3].slider_value);
    }

    #[test]
    fn low_frequency_bass_resolution_10_to_125_hz() {
        const RATE: u32 = 48_000;
        // 20 Hz tone
        let s20 = generate_sine(20.0, 1.0, 0.2, RATE);
        let spec20 = SpectrumAnalysis::from_mono(&s20, RATE);

        let sub_bass = spec20.band(10.0, 40.0);
        let mid_bass = spec20.band(60.0, 125.0);

        assert!((sub_bass.slider_value - 100.0).abs() < 3.0, "20 Hz in sub-bass band expected ~100.0, got {}", sub_bass.slider_value);
        assert!(mid_bass.slider_value < 1.0, "20 Hz in 60-125 Hz band should be rejected, got {}", mid_bass.slider_value);

        // 80 Hz tone
        let s80 = generate_sine(80.0, 1.0, 0.2, RATE);
        let spec80 = SpectrumAnalysis::from_mono(&s80, RATE);

        let sub_bass_80 = spec80.band(10.0, 40.0);
        let mid_bass_80 = spec80.band(60.0, 125.0);

        assert!(sub_bass_80.slider_value < 1.0, "80 Hz in 10-40 Hz band should be rejected, got {}", sub_bass_80.slider_value);
        assert!((mid_bass_80.slider_value - 100.0).abs() < 2.0, "80 Hz in 60-125 Hz band expected ~100.0, got {}", mid_bass_80.slider_value);
    }

    #[test]
    fn silence_and_invalid_inputs() {
        let spec_empty = SpectrumAnalysis::from_mono(&[], 48_000);
        assert_eq!(spec_empty.band(20.0, 200.0).slider_value, 0.0);

        let silence = vec![0.0f64; 1000];
        let spec_silence = SpectrumAnalysis::from_mono(&silence, 48_000);
        assert_eq!(spec_silence.band(20.0, 200.0).slider_value, 0.0);

        let s = generate_sine(100.0, 1.0, 0.1, 48_000);
        let spec = SpectrumAnalysis::from_mono(&s, 48_000);

        // Inverted range
        let out_inv = spec.band(250.0, 30.0);
        let out_norm = spec.band(30.0, 250.0);
        assert_eq!(out_inv, out_norm);

        // Negative and above Nyquist
        let out_neg = spec.band(-100.0, 250.0);
        assert!((out_neg.slider_value - 100.0).abs() < 2.0);

        let out_high = spec.band(30000.0, 40000.0);
        assert_eq!(out_high.slider_value, 0.0);

        // Zero rate
        let spec_zero_rate = SpectrumAnalysis::from_mono(&s, 0);
        assert_eq!(spec_zero_rate.band(30.0, 250.0).slider_value, 0.0);
    }

    #[test]
    fn stereo_downmix_and_channel_selection() {
        const RATE: u32 = 48_000;
        let count = (0.2 * RATE as f64) as usize;
        let mut stereo = Vec::with_capacity(count * 2);
        for i in 0..count {
            let t = i as f64 / RATE as f64;
            let l = (2.0 * PI * 100.0 * t).sin() as f32; // 100 Hz left
            let r = 0.0f32; // silence right
            stereo.push(l);
            stereo.push(r);
        }

        let spec_mono = SpectrumAnalysis::from_stereo_interleaved(&stereo, RATE, ChannelMode::Mono);
        let spec_l = SpectrumAnalysis::from_stereo_interleaved(&stereo, RATE, ChannelMode::Left);
        let spec_r = SpectrumAnalysis::from_stereo_interleaved(&stereo, RATE, ChannelMode::Right);

        let out_mono = spec_mono.band(30.0, 250.0);
        let out_l = spec_l.band(30.0, 250.0);
        let out_r = spec_r.band(30.0, 250.0);

        // Downmixing 1.0 left + 0.0 right gives 0.5 peak amplitude -> 50.0 slider value.
        assert!((out_mono.slider_value - 50.0).abs() < 1.5, "mono expected ~50.0, got {}", out_mono.slider_value);
        assert!((out_l.slider_value - 100.0).abs() < 2.0, "left expected ~100.0, got {}", out_l.slider_value);
        assert!(out_r.slider_value < 0.1, "right expected ~0.0, got {}", out_r.slider_value);
    }

    #[test]
    fn different_sample_rates_and_window_lengths() {
        for &rate in &[44_100, 48_000, 96_000] {
            for &duration_s in &[0.05, 0.1, 0.2] {
                let samples = generate_sine(440.0, 1.0, duration_s, rate);
                let spec = SpectrumAnalysis::from_mono(&samples, rate);
                let out = spec.band(400.0, 480.0);
                assert!(
                    (out.slider_value - 100.0).abs() < 3.0,
                    "rate {rate}, duration {duration_s}: expected ~100.0, got {}",
                    out.slider_value
                );
            }
        }
    }

    #[test]
    fn stateful_frequency_analyzer_consistency() {
        const RATE: u32 = 48_000;
        let mut analyzer = FrequencyAnalyzer::new();

        let s1 = generate_sine(100.0, 0.8, 0.2, RATE);
        let s2 = generate_sine(500.0, 0.3, 0.2, RATE);

        // Run 1: s1
        let a1 = analyzer.analyze_mono(&s1, RATE);
        assert!((a1.band(50.0, 150.0).slider_value - 80.0).abs() < 2.0);

        // Run 2: s2
        let a2 = analyzer.analyze_mono(&s2, RATE);
        assert!((a2.band(450.0, 550.0).slider_value - 30.0).abs() < 2.0);

        // Run 3: s1 again - verify no stale buffer contamination
        let a3 = analyzer.analyze_mono(&s1, RATE);
        assert_eq!(a1.band(50.0, 150.0).slider_value, a3.band(50.0, 150.0).slider_value);
    }
}
