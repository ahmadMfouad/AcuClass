pub mod audio_processor;
pub mod feature_extractor;
pub mod model_inference;
pub mod scaler;
pub mod stream_processor;

pub use audio_processor::AudioProcessor;
pub use feature_extractor::MfccExtractor;
pub use model_inference::NoiseClassifier;
pub use scaler::StandardScaler;
pub use stream_processor::StreamProcessor;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseLevel {
    Low = 0,
    Mid = 1,
    High = 2,
}

impl NoiseLevel {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(NoiseLevel::Low),
            1 => Some(NoiseLevel::Mid),
            2 => Some(NoiseLevel::High),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            NoiseLevel::Low => "Low Noise",
            NoiseLevel::Mid => "Mid Noise",
            NoiseLevel::High => "High Noise",
        }
    }
}

pub type Result<T> = anyhow::Result<T>;
