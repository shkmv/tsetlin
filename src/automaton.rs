//! Tsetlin Automaton implementation
//!
//! A Tsetlin automaton is a two-state finite automaton that learns to perform
//! binary actions based on rewards and penalties.

use rand::Rng;

/// Tsetlin Automaton state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Include the literal in the clause
    Include,
    /// Exclude the literal from the clause
    Exclude,
}

/// A Tsetlin Automaton that learns to include or exclude literals
#[derive(Debug, Clone)]
pub struct TsetlinAutomaton {
    /// Current state counter (1 to N for Include, -(1 to N) for Exclude)
    state: i32,
    /// Number of states in each action
    num_states: u32,
}

impl TsetlinAutomaton {
    /// Create a new Tsetlin automaton with the specified number of states
    pub fn new(num_states: u32) -> Self {
        Self {
            state: -(num_states as i32), // Start in deepest Exclude state
            num_states,
        }
    }

    /// Get the current action of the automaton
    pub fn action(&self) -> Action {
        if self.state > 0 {
            Action::Include
        } else {
            Action::Exclude
        }
    }

    /// Reward the automaton (reinforce current action)
    pub fn reward(&mut self) {
        if self.state > 0 {
            // In Include state, move towards deeper Include
            self.state = std::cmp::min(self.state + 1, self.num_states as i32);
        } else {
            // In Exclude state, move towards deeper Exclude
            self.state = std::cmp::max(self.state - 1, -(self.num_states as i32));
        }
    }

    /// Penalize the automaton (discourage current action)
    pub fn penalize(&mut self) {
        if self.state > 0 {
            // In Include state, move towards Exclude
            self.state -= 1;
        } else {
            // In Exclude state, move towards Include
            self.state += 1;
        }
    }

    /// Update the automaton with probability
    pub fn update_with_probability<R: Rng>(
        &mut self,
        reward: bool,
        probability: f64,
        rng: &mut R,
    ) {
        if rng.gen::<f64>() < probability {
            if reward {
                self.reward();
            } else {
                self.penalize();
            }
        }
    }
}

impl Default for TsetlinAutomaton {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_new_automaton() {
        let automaton = TsetlinAutomaton::new(100);
        assert_eq!(automaton.action(), Action::Exclude);
    }

    #[test]
    fn test_reward_and_penalize() {
        let mut automaton = TsetlinAutomaton::new(10);
        
        // Start in exclude state, penalize should move towards include
        automaton.penalize();
        assert_eq!(automaton.action(), Action::Exclude);
        
        // Continue penalizing to reach include state
        for _ in 0..10 {
            automaton.penalize();
        }
        assert_eq!(automaton.action(), Action::Include);
        
        // Now reward should move deeper into include
        automaton.reward();
        assert_eq!(automaton.action(), Action::Include);
    }

    #[test]
    fn test_update_with_probability() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut automaton = TsetlinAutomaton::new(100);
        let initial_state = automaton.state;
        
        // With probability 0.0, should never update
        for _ in 0..10 {
            automaton.update_with_probability(true, 0.0, &mut rng);
        }
        assert_eq!(automaton.state, initial_state);
        
        // With probability 1.0, should always update
        automaton.penalize(); // Move away from boundary
        let new_state = automaton.state;
        automaton.update_with_probability(true, 1.0, &mut rng);
        assert_ne!(automaton.state, new_state);
    }
}