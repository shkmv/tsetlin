//! Clause implementation for Tsetlin Machines
//!
//! A clause is a conjunction of literals that can be positive or negative features.

use crate::automaton::{Action, TsetlinAutomaton};
use rand::Rng;

/// Represents a single clause in a Tsetlin machine
#[derive(Debug, Clone)]
pub struct Clause {
    /// Automata for positive literals (one per feature)
    positive_automata: Vec<TsetlinAutomaton>,
    /// Automata for negative literals (one per feature)
    negative_automata: Vec<TsetlinAutomaton>,
}

impl Clause {
    /// Create a new clause with the specified number of features
    pub fn new(num_features: usize, num_states: u32) -> Self {
        Self {
            positive_automata: vec![TsetlinAutomaton::new(num_states); num_features],
            negative_automata: vec![TsetlinAutomaton::new(num_states); num_features],
        }
    }

    /// Evaluate the clause for a given input
    pub fn evaluate(&self, input: &[bool]) -> bool {
        for i in 0..input.len() {
            // Check positive literals
            if self.positive_automata[i].action() == Action::Include && !input[i] {
                return false;
            }
            
            // Check negative literals
            if self.negative_automata[i].action() == Action::Include && input[i] {
                return false;
            }
        }
        true
    }

    /// Update the clause based on feedback
    pub fn update<R: Rng>(
        &mut self,
        input: &[bool],
        target: bool,
        clause_output: bool,
        specificity: f64,
        rng: &mut R,
    ) {
        if target {
            // Type I feedback (positive target)
            if clause_output {
                // Clause fired correctly, reward included literals
                for i in 0..input.len() {
                    if self.positive_automata[i].action() == Action::Include {
                        if input[i] {
                            self.positive_automata[i].reward();
                        } else {
                            self.positive_automata[i].penalize();
                        }
                    }
                    
                    if self.negative_automata[i].action() == Action::Include {
                        if !input[i] {
                            self.negative_automata[i].reward();
                        } else {
                            self.negative_automata[i].penalize();
                        }
                    }
                }
            } else {
                // Clause didn't fire, include more literals with probability
                for i in 0..input.len() {
                    if self.positive_automata[i].action() == Action::Exclude && input[i] {
                        self.positive_automata[i].update_with_probability(
                            false,
                            specificity / (specificity + 1.0),
                            rng,
                        );
                    }
                    
                    if self.negative_automata[i].action() == Action::Exclude && !input[i] {
                        self.negative_automata[i].update_with_probability(
                            false,
                            specificity / (specificity + 1.0),
                            rng,
                        );
                    }
                }
            }
        } else {
            // Type II feedback (negative target)
            if clause_output {
                // Clause fired incorrectly, penalize randomly
                for i in 0..input.len() {
                    if self.positive_automata[i].action() == Action::Include {
                        self.positive_automata[i].update_with_probability(
                            false,
                            1.0 / specificity,
                            rng,
                        );
                    }
                    
                    if self.negative_automata[i].action() == Action::Include {
                        self.negative_automata[i].update_with_probability(
                            false,
                            1.0 / specificity,
                            rng,
                        );
                    }
                }
            }
        }
    }
}

/// A collection of clauses that vote on the final decision
#[derive(Debug, Clone)]
pub struct ClauseBank {
    /// All clauses in the bank
    clauses: Vec<Clause>,
    /// Polarity of each clause (true for positive, false for negative)
    polarities: Vec<bool>,
}

impl ClauseBank {
    /// Create a new clause bank
    pub fn new(num_features: usize, num_clauses: usize, num_states: u32) -> Self {
        let clauses = (0..num_clauses)
            .map(|_| Clause::new(num_features, num_states))
            .collect();
            
        let polarities = (0..num_clauses)
            .map(|i| i < num_clauses / 2) // First half positive, second half negative
            .collect();
            
        Self {
            clauses,
            polarities,
        }
    }

    /// Evaluate all clauses and return the vote sum
    pub fn vote(&self, input: &[bool]) -> i32 {
        let mut vote_sum = 0;
        
        for (clause, &polarity) in self.clauses.iter().zip(self.polarities.iter()) {
            if clause.evaluate(input) {
                if polarity {
                    vote_sum += 1;
                } else {
                    vote_sum -= 1;
                }
            }
        }
        
        vote_sum
    }

    /// Update all clauses based on feedback
    pub fn update<R: Rng>(
        &mut self,
        input: &[bool],
        target: bool,
        threshold: f64,
        specificity: f64,
        rng: &mut R,
    ) {
        let vote_sum = self.vote(input);
        
        for (clause, &polarity) in self.clauses.iter_mut().zip(self.polarities.iter()) {
            let clause_output = clause.evaluate(input);
            
            let should_update = if target {
                vote_sum < threshold as i32
            } else {
                vote_sum > -(threshold as i32)
            };
            
            if should_update {
                let clause_target = if polarity { target } else { !target };
                clause.update(input, clause_target, clause_output, specificity, rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_clause_evaluation() {
        let clause = Clause::new(3, 100);
        let input = vec![true, false, true];
        
        // Empty clause should always return true
        assert!(clause.evaluate(&input));
    }

    #[test]
    fn test_clause_bank_voting() {
        let bank = ClauseBank::new(3, 4, 100);
        let input = vec![true, false, true];
        
        // With empty clauses, all should fire
        // 2 positive clauses (+2) + 2 negative clauses (-2) = 0
        assert_eq!(bank.vote(&input), 0);
    }

    #[test]
    fn test_clause_bank_update() {
        let mut bank = ClauseBank::new(3, 4, 100);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let input = vec![true, false, true];
        
        // Update with positive target
        bank.update(&input, true, 1.0, 2.0, &mut rng);
        
        // Should not crash
        assert_eq!(bank.clauses.len(), 4);
    }
}