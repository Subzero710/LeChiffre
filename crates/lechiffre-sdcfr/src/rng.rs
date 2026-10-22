//! Versioned, serializable SplitMix64 stream; not intended for cryptography.
#[derive(Clone, Debug, PartialEq)]
pub struct Rng {
    pub(crate) state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
    pub fn below(&mut self, upper: u64) -> u64 {
        assert!(upper > 0);
        // Rejection removes modulo bias, including when sampling a reservoir slot.
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let n = self.next_u64();
            if n >= threshold {
                return n % upper;
            }
        }
    }
    pub fn action(&mut self, probabilities: &[f32]) -> usize {
        let threshold = self.unit();
        let mut cumulative = 0.0;
        let mut last = None;
        for (a, &p) in probabilities.iter().enumerate() {
            assert!(p.is_finite() && p >= 0.0);
            if p > 0.0 {
                last = Some(a);
            }
            cumulative += p as f64;
            if threshold < cumulative {
                return a;
            }
        }
        last.expect("cannot sample an empty policy")
    }
}
