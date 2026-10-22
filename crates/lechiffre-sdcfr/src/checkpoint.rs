//! Complete, versioned checkpoints at iteration boundaries. The optimizer is
//! deliberately reinitialized at each player update, so no optimizer is live here.
use crate::{
    game::Observation,
    memory::{Reservoir, Sample},
    neural::Mlp,
    rng::Rng,
    strategy::Snapshot,
    training::{Config, Trainer},
};
use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"LCSDCF01";
const MAX_BYTES: u64 = 64 * 1024 * 1024;
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

struct Encoder {
    bytes: Vec<u8>,
}
impl Encoder {
    fn integer(&mut self, n: u64) {
        self.bytes.extend(n.to_le_bytes());
    }
    fn float(&mut self, n: f32) {
        self.bytes.extend(n.to_le_bytes());
    }
    fn floats(&mut self, v: &[f32]) {
        self.integer(v.len() as u64);
        for &n in v {
            self.float(n);
        }
    }
    fn model(&mut self, m: &Mlp) {
        self.integer(m.sizes.len() as u64);
        for &n in &m.sizes {
            self.integer(n as u64);
        }
        self.floats(&m.parameters);
    }
    fn sample(&mut self, s: &Sample) {
        self.integer(s.iteration);
        self.floats(&s.observation.features);
        self.integer(s.observation.legal.len() as u64);
        self.bytes
            .extend(s.observation.legal.iter().map(|&b| u8::from(b)));
        self.floats(&s.targets);
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Decoder<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(n)
            .ok_or_else(|| invalid("length overflow"))?;
        let part = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid("truncated checkpoint"))?;
        self.position = end;
        Ok(part)
    }
    fn integer(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn length(&mut self, min_bytes: usize) -> io::Result<usize> {
        let n = usize::try_from(self.integer()?).map_err(|_| invalid("length overflow"))?;
        if n > (self.bytes.len() - self.position) / min_bytes {
            return Err(invalid("invalid collection length"));
        }
        Ok(n)
    }
    fn size(&mut self) -> io::Result<usize> {
        usize::try_from(self.integer()?).map_err(|_| invalid("size overflow"))
    }
    fn float(&mut self) -> io::Result<f32> {
        let x = f32::from_le_bytes(self.take(4)?.try_into().unwrap());
        if !x.is_finite() {
            return Err(invalid("non-finite checkpoint value"));
        }
        Ok(x)
    }
    fn floats(&mut self) -> io::Result<Vec<f32>> {
        let n = self.length(4)?;
        (0..n).map(|_| self.float()).collect()
    }
    fn model(&mut self) -> io::Result<Mlp> {
        let n = self.length(8)?;
        if !(2..=16).contains(&n) {
            return Err(invalid("invalid MLP depth"));
        }
        let sizes = (0..n)
            .map(|_| self.size())
            .collect::<io::Result<Vec<_>>>()?;
        let count = sizes
            .windows(2)
            .try_fold(0_usize, |n, s| {
                n.checked_add(s[0].checked_mul(s[1])?.checked_add(s[1])?)
            })
            .ok_or_else(|| invalid("network size overflow"))?;
        let parameters = self.floats()?;
        if sizes.contains(&0) || parameters.len() != count {
            return Err(invalid("invalid MLP shape"));
        }
        Ok(Mlp { sizes, parameters })
    }
    fn sample(&mut self) -> io::Result<Sample> {
        let iteration = self.integer()?;
        let features = self.floats()?;
        let n = self.length(1)?;
        let legal = self
            .take(n)?
            .iter()
            .map(|&b| match b {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(invalid("invalid action mask")),
            })
            .collect::<io::Result<Vec<_>>>()?;
        let targets = self.floats()?;
        Ok(Sample {
            observation: Observation { features, legal },
            targets,
            iteration,
        })
    }
}

impl Trainer {
    /// Refuses to overwrite an existing checkpoint; saves through a sibling temp file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.validate_checkpoint()?;
        let path = path.as_ref();
        if path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "checkpoint exists",
            ));
        }
        let mut e = Encoder {
            bytes: MAGIC.to_vec(),
        };
        e.integer(self.game_id.len() as u64);
        e.bytes.extend(self.game_id.as_bytes());
        let c = &self.config;
        for n in [
            c.traversals,
            c.train_steps,
            c.batch_size,
            c.reservoir_capacity,
            c.hidden,
        ] {
            e.integer(n as u64);
        }
        e.float(c.learning_rate);
        e.integer(c.seed);
        e.integer(self.completed);
        e.integer(self.rng.state);
        for player in 0..2 {
            e.model(&self.models[player]);
            let memory = &self.memory[player];
            e.integer(memory.seen);
            e.integer(memory.samples.len() as u64);
            for sample in &memory.samples {
                e.sample(sample);
            }
            e.integer(self.archive[player].len() as u64);
            for snapshot in &self.archive[player] {
                e.integer(snapshot.iteration);
                e.model(&snapshot.model);
            }
        }
        if e.bytes.len() as u64 > MAX_BYTES {
            return Err(invalid("reference checkpoint exceeds 64 MiB"));
        }
        let temp = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let result = (|| {
            file.write_all(&e.bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temp, path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }

    pub fn load(path: impl AsRef<Path>, expected_game_id: &str) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err(invalid("checkpoint exceeds 64 MiB"));
        }
        let mut d = Decoder {
            bytes: &bytes,
            position: 0,
        };
        if d.take(8)? != MAGIC {
            return Err(invalid("unknown checkpoint version"));
        }
        let n = d.length(1)?;
        let game_id =
            String::from_utf8(d.take(n)?.to_vec()).map_err(|_| invalid("invalid game id"))?;
        if game_id != expected_game_id {
            return Err(invalid("incompatible game/encoder/actions"));
        }
        let config = Config {
            traversals: d.size()?,
            train_steps: d.size()?,
            batch_size: d.size()?,
            reservoir_capacity: d.size()?,
            hidden: d.size()?,
            learning_rate: d.float()?,
            seed: d.integer()?,
        };
        config.validate().map_err(|e| invalid(&e))?;
        let completed = d.integer()?;
        let rng = Rng {
            state: d.integer()?,
        };
        let mut models = Vec::new();
        let mut memories = Vec::new();
        let mut archives = Vec::new();
        for _ in 0..2 {
            models.push(d.model()?);
            let seen = d.integer()?;
            let n = d.length(32)?;
            let samples = (0..n).map(|_| d.sample()).collect::<io::Result<Vec<_>>>()?;
            memories.push(Reservoir {
                capacity: config.reservoir_capacity,
                seen,
                samples,
            });
            let n = d.length(24)?;
            let archive = (0..n)
                .map(|_| {
                    Ok(Snapshot {
                        iteration: d.integer()?,
                        model: d.model()?,
                    })
                })
                .collect::<io::Result<Vec<_>>>()?;
            archives.push(archive);
        }
        if d.position != bytes.len() {
            return Err(invalid("unexpected trailing data"));
        }
        let trainer = Self {
            game_id,
            config,
            completed,
            rng,
            models: models.try_into().unwrap(),
            memory: memories.try_into().unwrap(),
            archive: archives.try_into().unwrap(),
        };
        trainer.validate_checkpoint()?;
        Ok(trainer)
    }

    fn validate_checkpoint(&self) -> io::Result<()> {
        self.config.validate().map_err(|e| invalid(&e))?;
        let shape = self.models[0].sizes();
        if shape.len() != 4 || shape[1] != self.config.hidden || shape[2] != self.config.hidden {
            return Err(invalid("incompatible architecture"));
        }
        let inputs = shape[0];
        let actions = shape[3];
        for player in 0..2 {
            if self.models[player].sizes() != shape
                || self.archive[player].len() as u64 != self.completed
            {
                return Err(invalid("inconsistent checkpoint iterations or models"));
            }
            for (index, snapshot) in self.archive[player].iter().enumerate() {
                if snapshot.iteration != index as u64 + 1 || snapshot.model.sizes() != shape {
                    return Err(invalid("inconsistent snapshot archive"));
                }
            }
            let memory = &self.memory[player];
            if memory.samples.len() as u64 != memory.seen.min(memory.capacity as u64) {
                return Err(invalid("inconsistent reservoir count"));
            }
            for sample in &memory.samples {
                if sample.iteration == 0
                    || sample.iteration > self.completed
                    || sample.observation.features.len() != inputs
                    || sample.targets.len() != actions
                    || sample.observation.legal.len() != actions
                    || !sample.observation.legal.contains(&true)
                    || sample
                        .targets
                        .iter()
                        .chain(&sample.observation.features)
                        .any(|x| !x.is_finite())
                {
                    return Err(invalid("inconsistent training sample"));
                }
            }
        }
        Ok(())
    }
}
