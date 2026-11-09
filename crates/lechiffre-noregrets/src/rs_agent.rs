//! NoRegrets as an ordinary rs_poker Agent. Only public actions are retained.
use crate::{bridge::card_to_noregrets, reference::UPSTREAM_COMMIT};
use async_trait::async_trait;
use noregrets::{
    abstraction::{AbsConfig, Abstraction, MenuShape},
    bot::Policy,
    cfr::{Blueprint, TrainConfig, Trainer},
    engine::{HandConfig, PlayerAction},
    table::Table,
};
use rand::{SeedableRng, rngs::SmallRng};
use rs_poker::arena::{
    Agent, GameState, Historian, HistorianError,
    action::{Action, AgentAction, PlayedActionPayload},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentMetadata {
    pub format: String,
    pub source: String,
    pub iterations: u64,
    pub seed: u64,
    #[serde(default = "heads_up")]
    pub num_players: usize,
    pub stack: u32,
    pub small_blind: u32,
    pub big_blind: u32,
    pub blueprint_sha256: String,
}

fn heads_up() -> usize {
    2
}

pub struct AgentReference {
    pub policy: Policy,
    pub metadata: AgentMetadata,
}

impl AgentReference {
    pub fn hand_config(&self) -> HandConfig {
        HandConfig {
            num_players: self.metadata.num_players,
            stack: self.metadata.stack,
            sb: self.metadata.small_blind,
            bb: self.metadata.big_blind,
        }
    }
    pub fn train(iterations: u64, seed: u64, stack: u32) -> Result<Self, String> {
        Self::train_for_players(iterations, seed, stack, 2)
    }
    pub fn train_for_players(
        iterations: u64,
        seed: u64,
        stack: u32,
        num_players: usize,
    ) -> Result<Self, String> {
        if iterations == 0 || stack <= 2 || !(2..=6).contains(&num_players) {
            return Err("positive iterations, stack > 2 and 2..6 players required".into());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .map_err(|e| e.to_string())?;
        pool.install(|| {
            let abs = Arc::new(Abstraction::new(AbsConfig {
                cache_cap: 200_000,
                equity_rollouts: 64,
                ..AbsConfig::default()
            }));
            let cfg = TrainConfig {
                hand: HandConfig {
                    num_players,
                    stack,
                    sb: 1,
                    bb: 2,
                },
                seed,
                prune_after: u64::MAX,
                ..TrainConfig::default()
            };
            let trainer = Trainer::new(abs.clone(), cfg);
            trainer.run(iterations, &|done| {
                eprintln!("NoRegrets Agent : {done}/{iterations}")
            });
            let bp = trainer.to_blueprint();
            Ok(Self {
                policy: Policy::new(bp, abs),
                metadata: AgentMetadata {
                    format: if num_players == 2 {
                        "noregrets-rs-agent-hu-unlimited-v1"
                    } else {
                        "noregrets-rs-agent-unlimited-v2"
                    }
                    .into(),
                    source: UPSTREAM_COMMIT.into(),
                    iterations,
                    seed,
                    num_players,
                    stack,
                    small_blind: 1,
                    big_blind: 2,
                    blueprint_sha256: String::new(),
                },
            })
        })
    }
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(p) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        fs::create_dir(path).map_err(|e| e.to_string())?;
        self.policy
            .blueprint
            .save(path.join("blueprint.bin").to_str().ok_or("invalid path")?)
            .map_err(|e| e.to_string())?;
        let mut m = self.metadata.clone();
        m.blueprint_sha256 = format!(
            "{:x}",
            Sha256::digest(fs::read(path.join("blueprint.bin")).map_err(|e| e.to_string())?)
        );
        fs::write(
            path.join("agent-reference.json"),
            serde_json::to_vec_pretty(&m).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let m: AgentMetadata =
            serde_json::from_slice(&fs::read(path.join("agent-reference.json")).map_err(|e| {
                format!("Native agent reference required (not the reduced SD-CFR reference): {e}")
            })?)
            .map_err(|e| e.to_string())?;
        let bytes = fs::read(path.join("blueprint.bin")).map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(bytes)) != m.blueprint_sha256 {
            return Err("blueprint SHA-256 mismatch".into());
        }
        let bp = Blueprint::load(path.join("blueprint.bin").to_str().ok_or("invalid path")?)
            .map_err(|e| e.to_string())?;
        let supported = (m.format == "noregrets-rs-agent-hu-unlimited-v1" && m.num_players == 2)
            || m.format == "noregrets-rs-agent-unlimited-v2";
        if !supported
            || !(2..=6).contains(&m.num_players)
            || m.source != UPSTREAM_COMMIT
            || bp.num_players != m.num_players
            || m.stack <= 2
            || m.small_blind != 1
            || m.big_blind != 2
            || bp.iterations != m.iterations
            || m.iterations == 0
            || !matches!(bp.abs_cfg.menu, MenuShape::Pluribus)
            || bp
                .strategies
                .values()
                .any(|s| s.is_empty() || s.iter().any(|p| !p.is_finite() || *p < 0.0))
        {
            return Err("incompatible native NoRegrets reference".into());
        }
        let abs = Arc::new(Abstraction::with_centroids(
            bp.abs_cfg.clone(),
            bp.centroids.clone(),
        ));
        Ok(Self {
            policy: Policy::new(bp, abs),
            metadata: m,
        })
    }
}

#[derive(Default, Clone)]
struct PublicLog {
    id: Option<u128>,
    actions: Vec<PlayedActionPayload>,
}
struct PublicHistorian(Arc<Mutex<PublicLog>>);
#[async_trait]
impl Historian for PublicHistorian {
    async fn record_action(
        &mut self,
        id: u128,
        _: &GameState,
        action: &Action,
    ) -> Result<(), HistorianError> {
        let mut log = self
            .0
            .lock()
            .map_err(|_| HistorianError::UnableToRecordAction)?;
        if matches!(action, Action::GameStart(_)) || log.id != Some(id) {
            log.id = Some(id);
            log.actions.clear();
        }
        match action {
            Action::PlayedAction(p) => log.actions.push(p.clone()),
            Action::FailedAction(p) => log.actions.push(p.result.clone()),
            _ => {}
        }
        Ok(())
    }
}

#[derive(Default, Debug, Clone, Serialize)]
pub struct AgentStats {
    pub decisions: u64,
    pub unseen: u64,
    pub decisions_after_off_tree_bet: u64,
    pub errors: Vec<String>,
}

pub struct NoRegretsAgent {
    reference: Arc<AgentReference>,
    name: String,
    rng: SmallRng,
    log: Arc<Mutex<PublicLog>>,
    stats: Arc<Mutex<AgentStats>>,
    last_id: Option<u128>,
    choices: HashMap<usize, noregrets::abstraction::AbsAction>,
}

impl NoRegretsAgent {
    pub fn new(reference: Arc<AgentReference>, seed: u64) -> Self {
        Self {
            reference,
            name: "NoRegrets".into(),
            rng: SmallRng::seed_from_u64(seed),
            log: Arc::new(Mutex::new(PublicLog::default())),
            stats: Arc::new(Mutex::new(AgentStats::default())),
            last_id: None,
            choices: HashMap::new(),
        }
    }
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    pub fn stats(&self) -> Arc<Mutex<AgentStats>> {
        self.stats.clone()
    }
    /// Fallible API for callers that want to handle setup/replay errors.
    /// Agent::act stops explicitly instead of silently playing a passive bot.
    pub fn try_action(&mut self, id: u128, gs: &GameState) -> Result<AgentAction, String> {
        let m = &self.reference.metadata;
        if gs.num_players != m.num_players
            || gs.starting_stacks.iter().any(|&s| s != m.stack as i64)
            || gs.big_blind != m.big_blind as i64
            || gs.small_blind != m.small_blind as i64
            || gs.ante != 0
            || gs.max_raises_per_round.is_some()
            || gs.rake.rate.numerator() != 0
        {
            return Err("NoRegrets reference needs matching player count, stacks/blinds, no rake and unlimited raises".into());
        }
        let log = self
            .log
            .lock()
            .map_err(|_| "public history lock poisoned")?
            .clone();
        if log.id != Some(id) {
            return Err("missing public history: use HoldemSimulationBuilder (registers Agent::historian automatically)".into());
        }
        if self.last_id != Some(id) {
            self.last_id = Some(id);
            self.choices.clear();
        }
        let action_count = log.actions.len();
        let hero = gs.to_act_idx();
        let seat = |p: usize| (p + gs.num_players - gs.dealer_idx) % gs.num_players;
        let mut deck = [u8::MAX; 52];
        let mut used = [false; 52];
        let hole: Vec<u8> = gs.hands[hero]
            .iter()
            .filter(|c| !gs.board.contains(c))
            .map(|c| card_to_noregrets(u8::from(c)))
            .collect();
        if hole.len() != 2 {
            return Err("expected exactly two private cards".into());
        }
        for (i, c) in hole.into_iter().enumerate() {
            deck[2 * seat(hero) + i] = c;
            used[c as usize] = true;
        }
        for (i, c) in gs.board.iter().enumerate() {
            let c = card_to_noregrets(u8::from(*c));
            deck[2 * gs.num_players + i] = c;
            used[c as usize] = true;
        }
        // Unknown opponent cards and future board are dummy unseen cards.
        // They affect no blueprint query, which only reads hero + public board.
        let mut rest = (0..52u8).filter(|c| !used[*c as usize]);
        for c in &mut deck {
            if *c == u8::MAX {
                *c = rest.next().ok_or("deck reconstruction failed")?;
            }
        }
        let abs = &self.reference.policy.abs;
        let mut table = Table::new(&self.reference.hand_config(), 0, deck);
        let mut off_tree = false;
        for (index, event) in log.actions.into_iter().enumerate() {
            if table.real.is_terminal() || table.real.to_act() != seat(event.idx) {
                return Err("public action replay diverged".into());
            }
            let action = if event.action == AgentAction::Fold {
                PlayerAction::Fold
            } else if event.final_bet > event.starting_bet {
                PlayerAction::RaiseTo(
                    u32::try_from(event.final_player_bet).map_err(|_| "bet out of range")?,
                )
            } else {
                PlayerAction::CheckCall
            };
            if let PlayerAction::RaiseTo(t) = action {
                let mapped = table.map_concrete(action, abs);
                if table.shadow.raise_bounds().is_some() {
                    off_tree |= abs.concrete(&table.shadow, mapped) != PlayerAction::RaiseTo(t);
                }
            }
            if let Some(&chosen) = self.choices.get(&index).filter(|_| event.idx == hero) {
                let actual = table.apply_abs(chosen, abs);
                if actual != action {
                    return Err("own action replay mismatch".into());
                }
            } else {
                table.apply_concrete(action, abs);
            }
        }
        if table.real.is_terminal()
            || table.real.to_act() != seat(hero)
            || table.real.pot() as i64 != gs.total_pot
            || table.real.current_bet() as i64 != gs.current_round_bet()
            || table.real.board()
                != gs
                    .board
                    .iter()
                    .map(|c| card_to_noregrets(u8::from(*c)))
                    .collect::<Vec<_>>()
        {
            return Err("rs_poker and NoRegrets public states diverged".into());
        }
        for p in 0..gs.num_players {
            if table.real.stack(seat(p)) as i64 != gs.stacks[p]
                || table.real.hand_commit(seat(p)) as i64 != gs.player_bet[p]
            {
                return Err("money replay diverged".into());
            }
        }
        let (actions, probs) =
            self.reference
                .policy
                .blueprint_dist(&table.shadow, &table.hist, &mut self.rng);
        let bucket = abs.bucket(
            table.shadow.hole(table.shadow.to_act()),
            table.shadow.board(),
            &mut self.rng,
        );
        let known = self
            .reference
            .policy
            .blueprint
            .get(bucket, &table.hist)
            .is_some_and(|s| s.len() == actions.len() && s.iter().sum::<f32>() > 0.0);
        let chosen = actions[noregrets::cfr::sample_index(&probs, &mut self.rng)];
        let raising = matches!(
            chosen,
            noregrets::abstraction::AbsAction::Bet(_)
                | noregrets::abstraction::AbsAction::AllIn
                | noregrets::abstraction::AbsAction::MinRaise
        );
        let concrete = if raising && table.real.raise_bounds().is_none() {
            PlayerAction::CheckCall
        } else {
            abs.concrete(&table.real, chosen)
        };
        let result = match concrete {
            PlayerAction::Fold => AgentAction::Fold,
            PlayerAction::CheckCall => AgentAction::Call,
            PlayerAction::RaiseTo(t) => AgentAction::Bet(t as i64),
        };
        if let AgentAction::Bet(t) = result {
            gs.validate_bet_amount(t)
                .map_err(|e| format!("invalid NoRegrets bet: {e:?}"))?;
        }
        self.choices.insert(action_count, chosen);
        let mut stats = self.stats.lock().map_err(|_| "statistics lock poisoned")?;
        stats.decisions += 1;
        stats.unseen += u64::from(!known);
        stats.decisions_after_off_tree_bet += u64::from(off_tree);
        Ok(result)
    }
}

#[async_trait]
impl Agent for NoRegretsAgent {
    async fn act(&mut self, id: u128, gs: &GameState) -> AgentAction {
        match self.try_action(id, gs) {
            Ok(a) => a,
            Err(e) => {
                self.stats
                    .lock()
                    .expect("stats poisoned")
                    .errors
                    .push(e.clone());
                panic!("NoRegrets Agent: {e}")
            }
        }
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn historian(&self) -> Option<Box<dyn Historian>> {
        Some(Box::new(PublicHistorian(self.log.clone())))
    }
}
