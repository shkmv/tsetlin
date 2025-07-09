# Tsetlin Machine Library for Rust

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple and efficient implementation of **Tsetlin Machines** in Rust. Tsetlin machines are interpretable machine learning algorithms that use propositional logic to learn patterns in data.

## 🚀 Features

- **Simple API**: Easy to use with minimal configuration
- **Efficient**: Optimized implementation using ndarray
- **Interpretable**: Logic-based machine learning
- **Pure Rust**: No external dependencies beyond ndarray and rand

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tsetlin = "0.1.0"
```

## 🏃 Quick Start

```rust
use tsetlin::TsetlinMachine;
use ndarray::{Array1, Array2};

// Create a simple XOR dataset
let features = Array2::from_shape_vec((4, 2), vec![
    true, false,   // [true, false] -> true
    false, true,   // [false, true] -> true
    true, true,    // [true, true] -> false
    false, false,  // [false, false] -> false
]).unwrap();
let labels = Array1::from_vec(vec![true, true, false, false]);

// Create and train a Tsetlin machine
let mut machine = TsetlinMachine::with_defaults(2, 20);
machine.fit(&features, &labels, 100);

// Make predictions
let predictions = machine.predict(&features);
let accuracy = machine.evaluate(&features, &labels);

println!("Accuracy: {:.2}", accuracy);
```

## 📝 API Reference

### TsetlinMachine

The main struct for creating and training Tsetlin machines.

#### Constructors

- `TsetlinMachine::new(num_features, num_clauses, specificity, threshold)` - Create with custom parameters
- `TsetlinMachine::with_defaults(num_features, num_clauses)` - Create with default parameters

#### Methods

- `fit(&mut self, features: &Array2<bool>, labels: &Array1<bool>, epochs: usize)` - Train the model
- `predict(&self, features: &Array2<bool>) -> Array1<bool>` - Make predictions on multiple samples
- `predict_single(&self, features: &[bool]) -> bool` - Make prediction on single sample
- `evaluate(&self, features: &Array2<bool>, labels: &Array1<bool>) -> f64` - Calculate accuracy

### Helper Functions

- `generate_xor_dataset()` - Generate XOR dataset for testing

## 🔬 Algorithm Overview

A Tsetlin machine consists of:

1. **Tsetlin Automata**: Two-state finite automata that learn to include or exclude literals
2. **Clauses**: Logical conjunctions of literals that evaluate to true or false
3. **Voting**: Clauses vote on the final decision (positive vs negative clauses)

The algorithm learns interpretable rules that can be analyzed and understood.

## 💡 Examples

### Basic Training

```rust
use tsetlin::TsetlinMachine;
use ndarray::{Array1, Array2};

// Create features and labels
let features = Array2::from_shape_vec((4, 2), vec![
    true, false, false, true, true, true, false, false
]).unwrap();
let labels = Array1::from_vec(vec![true, false, true, false]);

// Train model
let mut machine = TsetlinMachine::with_defaults(2, 20);
machine.fit(&features, &labels, 100);

// Evaluate
let accuracy = machine.evaluate(&features, &labels);
println!("Training accuracy: {:.2}", accuracy);
```

### Custom Parameters

```rust
use tsetlin::TsetlinMachine;

// Create with custom parameters
let mut machine = TsetlinMachine::new(
    10,    // num_features
    100,   // num_clauses
    3.0,   // specificity
    1.5    // threshold
);
```

### Using the XOR Helper

```rust
use tsetlin::{TsetlinMachine, generate_xor_dataset};

let (features, labels) = generate_xor_dataset();
let mut machine = TsetlinMachine::with_defaults(2, 40);
machine.fit(&features, &labels, 200);

let accuracy = machine.evaluate(&features, &labels);
println!("XOR accuracy: {:.2}", accuracy);
```

## 🧪 Testing

Run the test suite:

```bash
cargo test
```

## 📚 References

- [The Tsetlin Machine - A Game-Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic](https://arxiv.org/abs/1804.01508)
- [Tsetlin Machine Wiki](https://github.com/cair/TsetlinMachine/wiki)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
