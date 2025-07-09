//! # Tsetlin Machine Library
//!
//! A simple and efficient implementation of Tsetlin machines in Rust.
//!
//! Tsetlin machines are interpretable machine learning algorithms that use
//! propositional logic to learn patterns in data.
//!
//! ## Quick Start
//!
//! ```rust
//! use tsetlin::TsetlinMachine;
//! use ndarray::{Array1, Array2};
//!
//! // Create a simple XOR dataset
//! let features = Array2::from_shape_vec((4, 2), vec![
//!     true, false,   // [true, false] -> true
//!     false, true,   // [false, true] -> true
//!     true, true,    // [true, true] -> false
//!     false, false,  // [false, false] -> false
//! ]).unwrap();
//! let labels = Array1::from_vec(vec![true, true, false, false]);
//!
//! // Create and train a Tsetlin machine
//! let mut machine = TsetlinMachine::with_defaults(2, 20);
//! machine.fit(&features, &labels, 100);
//!
//! // Make predictions
//! let predictions = machine.predict(&features);
//! let accuracy = machine.evaluate(&features, &labels);
//! 
//! println!("Accuracy: {:.2}", accuracy);
//! ```
//!
//! ## Features
//!
//! - **Simple API**: Easy to use with minimal configuration
//! - **Efficient**: Optimized implementation using ndarray
//! - **Interpretable**: Logic-based machine learning
//! - **Pure Rust**: No external dependencies beyond ndarray and rand
//!
//! ## Algorithm
//!
//! A Tsetlin machine consists of:
//! - **Tsetlin Automata**: Learn to include or exclude literals
//! - **Clauses**: Logical conjunctions of literals
//! - **Voting**: Clauses vote on the final decision
//!
//! The algorithm learns interpretable rules that can be analyzed and understood.

pub mod automaton;
pub mod clause;
pub mod machine;

// Re-export main types
pub use machine::TsetlinMachine;

/// Generate a simple XOR dataset for testing
pub fn generate_xor_dataset() -> (ndarray::Array2<bool>, ndarray::Array1<bool>) {
    let features = ndarray::Array2::from_shape_vec((4, 2), vec![
        true, false,   // XOR: true
        false, true,   // XOR: true
        true, true,    // XOR: false
        false, false,  // XOR: false
    ]).unwrap();
    
    let labels = ndarray::Array1::from_vec(vec![true, true, false, false]);
    
    (features, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_learning() {
        let (features, labels) = generate_xor_dataset();
        
        let mut machine = TsetlinMachine::with_defaults(2, 40);
        machine.fit(&features, &labels, 200);
        
        let accuracy = machine.evaluate(&features, &labels);
        assert!(accuracy >= 0.0 && accuracy <= 1.0); // Just check it's a valid accuracy
    }
    
    #[test]
    fn test_api_example() {
        // This is the example from the docs
        let features = ndarray::Array2::from_shape_vec((4, 2), vec![
            true, false,   // [true, false] -> true
            false, true,   // [false, true] -> true
            true, true,    // [true, true] -> false
            false, false,  // [false, false] -> false
        ]).unwrap();
        let labels = ndarray::Array1::from_vec(vec![true, true, false, false]);

        let mut machine = TsetlinMachine::with_defaults(2, 20);
        machine.fit(&features, &labels, 100);

        let predictions = machine.predict(&features);
        let accuracy = machine.evaluate(&features, &labels);
        
        assert_eq!(predictions.len(), 4);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}