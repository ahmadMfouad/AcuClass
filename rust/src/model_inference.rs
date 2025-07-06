use ndarray::{Array1, Array2, ArrayD};
use ort::{
    session::{Session, builder::{SessionBuilder, GraphOptimizationLevel}}, 
    value::Value
};
use crate::{Result, NoiseLevel};

pub struct NoiseClassifier {
    session: Session,
}

impl NoiseClassifier {    pub fn new(model_path: &str) -> Result<Self> {
        // For ORT 2.0, try creating session without explicit environment
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        println!("Model loaded successfully!");
        println!("Input count: {}", session.inputs.len());
        if !session.inputs.is_empty() {
            println!("Input name: {}", session.inputs[0].name);
        }
        println!("Output count: {}", session.outputs.len());
        if !session.outputs.is_empty() {
            println!("Output name: {}", session.outputs[0].name);
        }

        Ok(Self { 
            session
        })
    }

    pub fn predict_batch(&mut self, features: &Array2<f64>) -> Result<Vec<NoiseLevel>> {
        let batch_size = features.nrows();
        
        // Convert f64 to f32 for ONNX (most models expect f32)
        let features_f32: Array2<f32> = features.mapv(|x| x as f32);
        
        // Create input tensor (ORT 2.0 style)
        let shape = features_f32.shape().to_vec();
        let data = features_f32.into_raw_vec();
        let input_tensor = Value::from_array((shape, data))?;
        
        // Run inference (ORT 2.0 style)
        let outputs = self.session.run(ort::inputs![input_tensor])?;
        
        // Extract predictions from the first output
        let output = &outputs[0];
        
        // Try to extract as different possible formats (ORT 2.0 style)
        let predictions = if let Ok(predictions_f32) = output.try_extract_tensor::<f32>() {
            // Single dimension output (direct class predictions)
            let (shape, data) = predictions_f32;
            let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let predictions_array = ArrayD::from_shape_vec(shape_vec, data.to_vec())?;
            Self::extract_predictions_from_1d(&predictions_array)?
        } else if let Ok(predictions_i64) = output.try_extract_tensor::<i64>() {
            // Integer predictions
            let (shape, data) = predictions_i64;
            let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let predictions_array = ArrayD::from_shape_vec(shape_vec, data.to_vec())?;
            Self::extract_predictions_from_1d_i64(&predictions_array)?
        } else {
            return Err(anyhow::anyhow!("Unsupported output tensor type"));
        };

        if predictions.len() != batch_size {
            return Err(anyhow::anyhow!(
                "Prediction count mismatch: expected {}, got {}", 
                batch_size, 
                predictions.len()
            ));
        }

        Ok(predictions)
    }

    fn extract_predictions_from_1d(predictions_array: &ArrayD<f32>) -> Result<Vec<NoiseLevel>> {
        let mut predictions = Vec::new();
        
        if predictions_array.ndim() == 1 {
            // Direct class predictions
            for &pred in predictions_array.iter() {
                let class_id = pred.round() as u8;
                if let Some(noise_level) = NoiseLevel::from_u8(class_id) {
                    predictions.push(noise_level);
                } else {
                    predictions.push(NoiseLevel::High); // Default fallback
                }
            }
        } else if predictions_array.ndim() == 2 {
            // Probability distributions - take argmax
            let predictions_2d = predictions_array.view().into_dimensionality::<ndarray::Ix2>()?;
            for row in predictions_2d.rows() {
                let max_idx = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(2); // Default to High Noise
                
                if let Some(noise_level) = NoiseLevel::from_u8(max_idx as u8) {
                    predictions.push(noise_level);
                } else {
                    predictions.push(NoiseLevel::High);
                }
            }
        } else {
            return Err(anyhow::anyhow!("Unexpected output tensor shape: {:?}", predictions_array.shape()));
        }
        
        Ok(predictions)
    }

    fn extract_predictions_from_1d_i64(predictions_array: &ArrayD<i64>) -> Result<Vec<NoiseLevel>> {
        let mut predictions = Vec::new();
        
        if predictions_array.ndim() == 1 {
            // Direct class predictions
            for &pred in predictions_array.iter() {
                let class_id = pred as u8;
                if let Some(noise_level) = NoiseLevel::from_u8(class_id) {
                    predictions.push(noise_level);
                } else {
                    predictions.push(NoiseLevel::High); // Default fallback
                }
            }
        } else if predictions_array.ndim() == 2 {
            // This case is less common for integer outputs, but handle it
            let predictions_2d = predictions_array.view().into_dimensionality::<ndarray::Ix2>()?;
            for row in predictions_2d.rows() {
                let max_idx = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.cmp(b))
                    .map(|(idx, _)| idx)
                    .unwrap_or(2); // Default to High Noise
                
                if let Some(noise_level) = NoiseLevel::from_u8(max_idx as u8) {
                    predictions.push(noise_level);
                } else {
                    predictions.push(NoiseLevel::High);
                }
            }
        } else {
            return Err(anyhow::anyhow!("Unexpected output tensor shape: {:?}", predictions_array.shape()));
        }
        
        Ok(predictions)
    }

    pub fn predict_single(&mut self, features: &Array1<f64>) -> Result<NoiseLevel> {
        // Reshape to batch format (1, feature_size)
        let batch_features = features.clone().insert_axis(ndarray::Axis(0));
        let predictions = self.predict_batch(&batch_features)?;
        
        predictions.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("No prediction returned"))
    }

    pub fn aggregate_predictions(&self, predictions: &[NoiseLevel]) -> Option<NoiseLevel> {
        if predictions.is_empty() {
            return None;
        }

        let mut counts = [0; 3];
        for &pred in predictions {
            counts[pred as usize] += 1;
        }

        let max_count = *counts.iter().max().unwrap();
        let most_common: Vec<usize> = counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count == max_count)
            .map(|(idx, _)| idx)
            .collect();

        if most_common.len() > 1 {
            // Tie case - default to High Noise for safety
            Some(NoiseLevel::High)
        } else {
            NoiseLevel::from_u8(most_common[0] as u8)
        }
    }
}
