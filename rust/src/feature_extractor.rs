use ndarray::{Array1, Array2};
use rustfft::{FftPlanner, num_complex::Complex};
use crate::Result;

pub struct MfccExtractor {
    sample_rate: f32,
    n_mfcc: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
}

impl MfccExtractor {
    pub fn new(sample_rate: f32, n_mfcc: usize) -> Self {
        let hop_length = (0.025 * sample_rate) as usize; // 25ms frame length
        
        Self {
            sample_rate,
            n_mfcc,
            n_fft: 2048,
            hop_length,
            n_mels: 128,
        }
    }

    pub fn extract_features(&self, audio: &[f32]) -> Result<Array1<f64>> {
        let mfccs = self.compute_mfcc(audio)?;
        
        // Compute mean and std for MFCCs (26 features total: 13 mean + 13 std)
        let mean_mfccs = mfccs.mean_axis(ndarray::Axis(1)).unwrap();
        let std_mfccs = mfccs.std_axis(ndarray::Axis(1), 0.0);
        
        // Concatenate mean and std
        let mut features = Vec::with_capacity(self.n_mfcc * 2);
        features.extend(mean_mfccs.iter().map(|&x| x as f64));
        features.extend(std_mfccs.iter().map(|&x| x as f64));
        
        // Clip outliers
        for val in &mut features {
            *val = val.clamp(-100.0, 100.0);
        }
        
        Ok(Array1::from(features))
    }

    fn compute_mfcc(&self, audio: &[f32]) -> Result<Array2<f32>> {
        // Compute power spectrogram
        let spectrogram = self.compute_spectrogram(audio)?;
        
        // Apply mel filter bank
        let mel_spectrogram = self.apply_mel_filters(&spectrogram)?;
        
        // Apply DCT to get MFCCs
        let mfccs = self.apply_dct(&mel_spectrogram)?;
        
        Ok(mfccs)
    }

    fn compute_spectrogram(&self, audio: &[f32]) -> Result<Array2<f32>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        
        let n_frames = (audio.len() - self.n_fft) / self.hop_length + 1;
        let mut spectrogram = Array2::zeros((self.n_fft / 2 + 1, n_frames));
        
        for (frame_idx, frame_start) in (0..audio.len() - self.n_fft + 1)
            .step_by(self.hop_length)
            .enumerate()
        {
            if frame_idx >= n_frames {
                break;
            }
            
            // Apply Hann window
            let mut windowed: Vec<Complex<f32>> = audio[frame_start..frame_start + self.n_fft]
                .iter()
                .enumerate()
                .map(|(i, &sample)| {
                    let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (self.n_fft - 1) as f32).cos());
                    Complex::new(sample * window, 0.0)
                })
                .collect();
            
            fft.process(&mut windowed);
            
            // Compute power spectrum
            for (i, &complex_val) in windowed.iter().take(self.n_fft / 2 + 1).enumerate() {
                spectrogram[[i, frame_idx]] = complex_val.norm_sqr();
            }
        }
        
        Ok(spectrogram)
    }

    fn apply_mel_filters(&self, spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        // Simplified mel filter bank implementation
        // In a production system, you'd want a more sophisticated implementation
        let mel_filters = self.create_mel_filters()?;
        let mel_spectrogram = mel_filters.dot(spectrogram);
        
        // Apply log
        let log_mel = mel_spectrogram.mapv(|x| (x + 1e-10).ln());
        
        Ok(log_mel)
    }

    fn create_mel_filters(&self) -> Result<Array2<f32>> {
        // Simplified mel filter bank creation
        let n_freqs = self.n_fft / 2 + 1;
        let mut filters = Array2::zeros((self.n_mels, n_freqs));
        
        // Create triangular filters (simplified implementation)
        let mel_low = self.hz_to_mel(0.0);
        let mel_high = self.hz_to_mel(self.sample_rate / 2.0);
        let mel_points: Vec<f32> = (0..=self.n_mels + 1)
            .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (self.n_mels + 1) as f32)
            .collect();
        
        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| self.mel_to_hz(mel)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((hz * self.n_fft as f32) / self.sample_rate).floor() as usize)
            .collect();
        
        for m in 0..self.n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];
            
            for k in left..=right {
                if k < n_freqs {
                    if k <= center {
                        filters[[m, k]] = (k - left) as f32 / (center - left) as f32;
                    } else {
                        filters[[m, k]] = (right - k) as f32 / (right - center) as f32;
                    }
                }
            }
        }
        
        Ok(filters)
    }

    fn apply_dct(&self, mel_spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        let (n_mels, n_frames) = mel_spectrogram.dim();
        let mut mfccs = Array2::zeros((self.n_mfcc, n_frames));
        
        for frame in 0..n_frames {
            for i in 0..self.n_mfcc {
                let mut sum = 0.0;
                for j in 0..n_mels {
                    sum += mel_spectrogram[[j, frame]] * 
                           (std::f32::consts::PI * i as f32 * (j as f32 + 0.5) / n_mels as f32).cos();
                }
                mfccs[[i, frame]] = sum;
            }
        }
        
        Ok(mfccs)
    }

    fn hz_to_mel(&self, hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(&self, mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }
}
