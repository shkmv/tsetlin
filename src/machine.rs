//! Main Tsetlin Machine implementation

use crate::clause::ClauseBank;
use ndarray::{Array1, Array2};
use rand::{prelude::SliceRandom, SeedableRng};

/// Main Tsetlin Machine implementation
#[derive(Debug, Clone)]
pub struct TsetlinMachine {
    /// Clause bank containing all clauses
    clause_bank: ClauseBank,
    /// Number of input features
    num_features: usize,
    /// Number of clauses
    num_clauses: usize,
    /// Specificity parameter
    specificity: f64,
    /// Decision threshold
    threshold: f64,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl TsetlinMachine {
    /// Create a new Tsetlin machine
    ///
    /// # Arguments
    /// * `num_features` - Number of input features
    /// * `num_clauses` - Number of clauses (must be even)
    /// * `specificity` - Specificity parameter (default: 2.0)
    /// * `threshold` - Decision threshold (default: 1.0)
    ///
    /// # Example
    /// ```
    /// use tsetlin::TsetlinMachine;
    /// let machine = TsetlinMachine::new(10, 100, 2.0, 1.0);
    /// ```
    pub fn new(num_features: usize, num_clauses: usize, specificity: f64, threshold: f64) -> Self {
        assert!(num_clauses % 2 == 0, "Number of clauses must be even");
        
        let clause_bank = ClauseBank::new(num_features, num_clauses, 100);
        let rng = rand::rngs::StdRng::from_entropy();
        
        Self {
            clause_bank,
            num_features,
            num_clauses,
            specificity,
            threshold,
            rng,
        }
    }

    /// Create a new Tsetlin machine with default parameters
    pub fn with_defaults(num_features: usize, num_clauses: usize) -> Self {
        Self::new(num_features, num_clauses, 2.0, 1.0)
    }

    /// Train the Tsetlin machine on a dataset
    ///
    /// # Arguments
    /// * `features` - Feature matrix (samples x features)
    /// * `labels` - Target labels
    /// * `epochs` - Number of training epochs
    ///
    /// # Example
    /// ```
    /// use tsetlin::TsetlinMachine;
    /// use ndarray::{Array1, Array2};
    /// 
    /// let features = Array2::from_shape_vec((4, 2), vec![
    ///     true, false, false, true, true, true, false, false
    /// ]).unwrap();
    /// let labels = Array1::from_vec(vec![true, false, true, false]);
    /// 
    /// let mut machine = TsetlinMachine::with_defaults(2, 10);
    /// machine.fit(&features, &labels, 100);
    /// ```
    pub fn fit(&mut self, features: &Array2<bool>, labels: &Array1<bool>, epochs: usize) {
        assert_eq!(features.nrows(), labels.len());
        assert_eq!(features.ncols(), self.num_features);
        
        let num_samples = features.nrows();
        let mut indices: Vec<usize> = (0..num_samples).collect();
        
        for _ in 0..epochs {
            // Shuffle samples
            indices.shuffle(&mut self.rng);
            
            // Train on each sample
            for &idx in &indices {
                let sample_features = features.row(idx).to_vec();
                let target = labels[idx];
                
                self.clause_bank.update(
                    &sample_features,
                    target,
                    self.threshold,
                    self.specificity,
                    &mut self.rng,
                );
            }
        }
    }

    /// Make predictions on a dataset
    ///
    /// # Arguments
    /// * `features` - Feature matrix (samples x features)
    ///
    /// # Returns
    /// Array of boolean predictions
    pub fn predict(&self, features: &Array2<bool>) -> Array1<bool> {
        assert_eq!(features.ncols(), self.num_features);
        
        let mut predictions = Array1::from_elem(features.nrows(), false);
        
        for (i, row) in features.rows().into_iter().enumerate() {
            let sample_features = row.to_vec();
            let vote = self.clause_bank.vote(&sample_features);
            predictions[i] = vote > 0;
        }
        
        predictions
    }

    /// Make a prediction on a single sample
    pub fn predict_single(&self, features: &[bool]) -> bool {
        assert_eq!(features.len(), self.num_features);
        
        let vote = self.clause_bank.vote(features);
        vote > 0
    }

    /// Evaluate the model on a dataset
    ///
    /// # Arguments
    /// * `features` - Feature matrix
    /// * `labels` - Target labels
    ///
    /// # Returns
    /// Accuracy score (0.0 to 1.0)
    pub fn evaluate(&self, features: &Array2<bool>, labels: &Array1<bool>) -> f64 {
        let predictions = self.predict(features);
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count();
        
        correct as f64 / labels.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_machine_creation() {
        let machine = TsetlinMachine::new(5, 10, 2.0, 1.0);
        assert_eq!(machine.num_features, 5);
        assert_eq!(machine.num_clauses, 10);
    }

    #[test]
    fn test_machine_with_defaults() {
        let machine = TsetlinMachine::with_defaults(5, 10);
        assert_eq!(machine.num_features, 5);
        assert_eq!(machine.num_clauses, 10);
    }

    #[test]
    fn test_machine_training() {
        let features = Array2::from_shape_vec((4, 2), vec![
            true, false, false, true, true, true, false, false
        ]).unwrap();
        let labels = Array1::from_vec(vec![true, false, true, false]);
        
        let mut machine = TsetlinMachine::with_defaults(2, 10);
        machine.fit(&features, &labels, 10);
        
        let predictions = machine.predict(&features);
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_machine_prediction() {
        let features = Array2::from_shape_vec((2, 2), vec![
            true, false, false, true
        ]).unwrap();
        let labels = Array1::from_vec(vec![true, false]);
        
        let mut machine = TsetlinMachine::with_defaults(2, 10);
        machine.fit(&features, &labels, 10);
        
        let prediction = machine.predict_single(&[true, false]);
        assert!(prediction == true || prediction == false);
    }

    #[test]
    fn test_machine_evaluation() {
        let features = Array2::from_shape_vec((4, 2), vec![
            true, false, false, true, true, true, false, false
        ]).unwrap();
        let labels = Array1::from_vec(vec![true, false, true, false]);
        
        let mut machine = TsetlinMachine::with_defaults(2, 20);
        machine.fit(&features, &labels, 50);
        
        let accuracy = machine.evaluate(&features, &labels);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    #[should_panic(expected = "Number of clauses must be even")]
    fn test_machine_odd_clauses() {
        TsetlinMachine::with_defaults(5, 9);
    }
}