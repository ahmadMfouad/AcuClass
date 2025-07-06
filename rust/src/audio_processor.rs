use hound::WavReader;
use std::path::Path;
use crate::Result;

pub struct AudioProcessor {
    target_sample_rate: u32,
}

impl AudioProcessor {
    pub fn new(target_sample_rate: u32) -> Self {
        Self { target_sample_rate }
    }

    pub fn load_audio_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();
    
        let audio_data = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.samples::<f32>()
                    .collect::<std::result::Result<Vec<f32>, _>>()
                    .map_err(|e| anyhow::anyhow!("Failed to read float samples: {}", e))?
            }
            hound::SampleFormat::Int => {
                reader.samples::<i32>()
                    .map(|s| s.map(|sample| sample as f32 / i32::MAX as f32))
                    .collect::<std::result::Result<Vec<f32>, _>>()
                    .map_err(|e| anyhow::anyhow!("Failed to read int samples: {}", e))?
            }
        };
    
        let mut processed_audio = audio_data;
    
        // Convert to mono if stereo
        if spec.channels == 2 {
            processed_audio = self.stereo_to_mono(&processed_audio);
        }
    
        // Resample if necessary
        if spec.sample_rate != self.target_sample_rate {
            processed_audio = self.resample(&processed_audio, spec.sample_rate, self.target_sample_rate)?;
        }
    
        Ok(processed_audio)
    }

    pub fn extract_chunks(&self, audio: &[f32], chunk_duration: f32) -> Vec<Vec<f32>> {
        let chunk_size = (chunk_duration * self.target_sample_rate as f32) as usize;
        let mut chunks = Vec::new();
        
        for i in (0..audio.len()).step_by(chunk_size) {
            let end = (i + chunk_size).min(audio.len());
            if end - i >= chunk_size {
                chunks.push(audio[i..end].to_vec());
            }
        }
        
        chunks
    }

    fn stereo_to_mono(&self, stereo: &[f32]) -> Vec<f32> {
        stereo.chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).unwrap_or(&0.0)) / 2.0)
            .collect()
    }

    fn resample(&self, audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        // Simple linear interpolation resampling
        // For production, consider using a proper resampling library
        let ratio = to_rate as f32 / from_rate as f32;
        let new_length = (audio.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);
        
        for i in 0..new_length {
            let src_index = i as f32 / ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = (src_index_floor + 1).min(audio.len() - 1);
            let fraction = src_index - src_index_floor as f32;
            
            let sample = if src_index_floor < audio.len() {
                audio[src_index_floor] * (1.0 - fraction) + 
                audio[src_index_ceil] * fraction
            } else {
                0.0
            };
            
            resampled.push(sample);
        }
        
        Ok(resampled)
    }
}
