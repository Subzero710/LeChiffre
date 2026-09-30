# Exact NLHE cash money and hand histories

This change covers the NLHE cash engine, its CFR integrations, OHH boundaries,
room text import, and normalized replay. Existing card evaluation, Omaha
evaluation, and the independent tournament ICM estimator retain their scope.

## Monetary representation

`rs_poker::Chips` is an `i64` alias, re-exported by `arena::Chips` and
`arena::money::Chips`. One unit is one cent at a currency boundary: `$0.01` is
`1`, `$1.00` is `100`, and `$123.45` is `12_345`. Existing tests that model
abstract chips keep their magnitudes; an abstract `100.0` becomes `100`.

Stacks, starting stacks, blinds, antes, bets, minimum raises, pots, refunds,
awards, rake, and raw monetary statistics use `Chips`. `AgentAction::Bet`
contains a street target in integer chips; OHH action amounts describe the
contribution made by that action. The converter and replay explicitly bridge
that distinction.

Calls, all-ins, zero stacks, and minimum raises use exact comparisons. Short
all-ins do not reopen betting for players whose previous action has not faced
a full raise. Short blinds preserve the nominal amount owed. The builder
checks that the starting ledger, including already posted contributions,
fits the signed integer range.

`money::apply_ratio` is the shared sizing boundary for pot fractions, raise
multipliers, and stack sizes expressed in big blinds. It multiplies the
supplied IEEE-754 `f32` value using integer mantissa/exponent arithmetic and
rounds to the nearest chip, with a half chip rounded upward. Invalid negative
or non-finite ratios are rejected. Oversized generated amounts saturate at
`Chips::MAX`; action generators then apply stack and legality limits.

## CFR boundaries

Action configurations retain floating sizing ratios. Generated monetary
actions and their deduplication use integer chips. Call and all-in index
comparisons are exact, including when both represent the same amount.

`ActionIndexMapper` uses local `f64` logarithmic interpolation. Its endpoints
are the exact configured chip amounts, and interior representatives round to
the nearest chip with ties upward before an integer clamp. Very large, nearby
chip amounts can share a logarithmic bin; the mapper is an abstraction and is
not an exact inverse for every integer amount. Effective-range doubling
saturates at the ledger limit.

Recursive and fast-forward simulations use the same pot and rake plan.
Individual showdown outcomes are calculated in `Chips` before conversion to
CFR utility. Board enumeration averages completed rewards in floating point.
The CFR reward retains its existing chip-unit scale; this migration does not
normalize the solver's utilities by the big blind or change regret semantics.

## Pot settlement and rake

`arena::pot` constructs main and side pots from integer contribution levels,
including contributions from folded players. Adjacent levels with identical
eligible winners are combined. Distinct eligibility sets remain separate.
A single contributor's unmatched excess is returned without rake.

Both normal simulation and CFR consume this shared plan. Every pot satisfies
`gross = net awards + rake`; after settlement, total starting stacks equal
final stacks plus hand rake. Uncalled refunds are removed from the contested
pot and are not counted as winnings.

Ties split whole chips. Remainders go clockwise among winners from the first
winning seat left of the button, separately for each pot. This engine policy
follows [Poker TDA rule 21(A)](https://www.pokertda.com/view-poker-tda-rules/).
The citation establishes the standard board-game rule; it does not establish
that every imported online room uses that exact odd-chip policy.

`RakeRate::new` validates a rational numerator/denominator: the denominator
must be positive and the numerator cannot exceed it. `RakeConfig` contains
the rate, an optional cap, a no-flop-no-drop flag, and explicit rounding.
`None` means unlimited; negative caps are rejected. `RakeConfig::none()` is
zero rake.

Each gross pot slice is multiplied in `i128`, rounded with integer
quotient/remainder arithmetic, then limited by the remaining hand cap and
the pot amount. Supported rounding policies are floor, ceil, and true
round-half-to-even. The cap is shared across all pots in the hand. For NLHE,
no-flop-no-drop permits rake only after three board cards have been dealt.

## Published platform schedules

The engine consumes `RakeConfig`; it does not branch on a room. The separate
`arena::rake::schedule` layer selects published data using platform, exact
stakes, dealt-player count, and table product. It is a fixed data snapshot,
verified against the linked official pages on 2026-09-30, not an automatic
fee update service or a currency converter.

| Platform | Implemented stakes/product | Rate and caps | Explicit policy limits |
| --- | --- | --- | --- |
| PokerStars | Published USD/EUR/GBP NLHE cash rows; regular and Zoom, including the special USD Zoom micro-stakes table | Published row-specific rates/caps; USD $100/$200+ uses the published high-stakes row | Published half-to-even and no-rake-before-flop behavior; non-USD caps remain a dated snapshot because PokerStars reviews them quarterly |
| CoinPoker | USDT NLHE regular and heads-up schedules through the published high/VIP rows | 5%; regular and HU caps are kept separate; BB-denominated high-stakes caps are resolved exactly | Exact base-rake rounding and no-flop-no-drop remain unverified; localized official pages currently disagree on the $2/$5 3–4 player cap, so that exact context returns an error |
| GGPoker | Published USD six-max rows through $10/$20 and nine-max rows through $5/$10 | 5%; separate caps for 2, 3, 4, and 5+ players, including the published BB-derived high-stakes caps | Exact rounding and no-flop-no-drop require caller input; Rush & Cash/promotional charges are excluded; nine-max antes are supplied separately |

Official sources:

- [PokerStars rake](https://www.pokerstars.com/poker/room/rake/)
- [CoinPoker fees (English)](https://coinpoker.com/rake/)
- [CoinPoker fees (French locale)](https://coinpoker.com/fr/rake/)
- [CoinPoker USDT poker](https://coinpoker.com/online-poker/tether/)
- [GGPoker six-max/nine-max NLHE information](https://legal.ggpoker.com/poker-games/texas-holdem/)

`RakeContext` includes an explicit currency/denomination. The schedule never
selects a USD row for EUR/GBP/USDT merely because the numerical blinds match.
CoinPoker cash play is keyed as USDT; its public fee tables display dollar-style
stake labels, while CoinPoker's current USDT material describes poker balances
and play in USDT. No exchange-rate conversion is performed.

CoinPoker splash charges and external cash drops are excluded from the base
rake model. GG jackpot, promotional, and Rush & Cash charges are also outside
these schedules. Unsupported stakes/products/currencies return
`ScheduleError::Unsupported`; a documented conflict between current official
CoinPoker locale pages returns `ScheduleError::ConflictingPublishedData` rather
than silently choosing one value.

`rake_config_for` returns an error if the schedule lacks a verified policy.
Callers can inspect `rake_schedule_for` and supply explicit missing policies
to `RakeSchedule::resolve`. Verified policies retain their published value.
Observed hand payouts never choose the replay's rake policy automatically.

## Exact OHH currency boundary

[OHH](https://hh-specs.handhistory.org/) remains the normalized hand-history
format. Its Rust monetary fields contain cents while its external JSON
numbers contain decimal currency units. `100 Chips` serializes as `1.00`,
and an existing JSON amount `1.06` deserializes as `106 Chips`.

`open_hand_history::amount` uses `serde_json` arbitrary-precision numeric
tokens and integer decimal parsing. It accepts exactly representable decimal
and exponent forms, rejects fractional cents and overflow, and does not
route monetary numbers through `f32` or `f64`. Currency still identifies the
denomination; no exchange-rate conversion occurs.

Missing/null amounts are accepted only for actions that do not move chips.
Monetary actions require an amount. Optional monetary fields preserve their
absence. Existing empty-string card conventions remain supported.

Tournament OHH remains parseable even though replay is cash-only. At the full
`HandHistory` boundary, table-chip fields (blinds, stacks, actions, pots and
wins) are normalized as integer tournament chips rather than cents; currency
fields in `tournament_info` (buy-in/fees/bounty fees) remain currency cents.
Older OHH examples that contain `tournament_info` but omit `tournament: true`
are normalized to tournament mode. Fractional tournament table chips are
rejected instead of being silently rounded.

The arena converter records integer actions, refunds, awards, and rake, checks
state consistency, and emits OHH-compatible one-based action numbers within each round.
Replay also accepts the branch's legacy zero-based numbering for backward compatibility.
Simulation → OHH JSON → OHH decode → replay tests cover both heads-up and
multiway settlement.

## Room text import

`PokerStarsParser`, `CoinPokerParser`, and `GGPokerParser` implement
`HandHistoryParser`. `RoomHandReader<R: BufRead>` yields one normalized hand at
a time and limits a single hand to 20,000 nonempty lines. Multiple consecutive
hands are supported without buffering an entire database.

The supported subset is English, single-board NLHE cash text with the
headers and monetary syntax represented by the restored fixtures. Parsers
capture identity, table/seats/button, stacks, blinds, uniform antes, known
cards, actions/all-ins, boards, collections, refunds, and aggregate rake.
Contribution levels determine main and side pots. Per-pot gross minus
reported winnings allocates the recorded aggregate rake; replay then checks
that allocation against an independently supplied policy.

| Room | Header/currency forms | Time handling |
| --- | --- | --- |
| PokerStars | `PokerStars Hand #...`, `Hold'em No Limit`, dollar amounts with USD | Explicit UTC/GMT or a supported fixed-offset abbreviation |
| CoinPoker | `CoinPoker Hand #...`, `NLH`, dollar or `₮` amounts; `RETURN` refunds | Explicit supported zone; both stake symbols normalize to the room's USDT game currency |
| GGPoker | `Poker Hand #...`, `Hold'em No Limit`, dollar amounts; known zero extra-fee summary fields | A caller-supplied fixed offset is required when the export omits its zone |

Supported fixed-offset labels are UTC/GMT, PDT/PST, and EDT/EST. A generic
`ET` label is rejected rather than inferring daylight-saving rules. The
existing GG fixture test explicitly supplies UTC as a test input; that is
not evidence that timezone-free PokerCraft exports are UTC.

Unknown action/monetary lines return `HandHistoryParseError` with room, hand
ID, source line, text, and reason. Overflowing summary winnings also return
this structured error. Tournaments, other variants, multiple runouts,
straddles/dead blinds, live seat/chip adjustments, cashouts, external splash
awards, and nonzero extra fees are outside the supported subset. Arbitrary
localized or revised room export forms are not claimed to be supported.

Fixture provenance is recorded conservatively:

| Fixture | Hand ID | Available evidence |
| --- | --- | --- |
| `tests/fixtures/rooms/pokerstars.txt` | `171562910425` | Restored from supplied cumulative checkpoints; earlier reports describe a researched room example |
| `tests/fixtures/rooms/coinpoker.txt` | `93840400001` | Same checkpoint lineage; contains a PDT timestamp and USDT amounts |
| `tests/fixtures/rooms/ggpoker.txt` | `RC3877234115` | Same checkpoint lineage; contains a Rush & Cash table name and no timestamp zone |
| `tests/fixtures/rooms/coinpoker_synthetic_sidepots.txt` | See fixture | Explicit synthetic multiway regression case, not an official room fee example |

The original published URLs for the first three fixtures were not preserved
in the provided checkpoints. Their independent provenance and any earlier
normalization edits are therefore unverified in this continuation. This
document does not label them unmodified official exports. Parsing the GG
fixture does not add a Rush & Cash schedule to the published-data layer.

## Replay and transitions

`open_hand_history::replay::replay_hand(hand, rake_config)` validates uncapped
NLHE cash OHH using `GameState` betting and the shared pot plan. It checks
players/seats/stacks, forced bets, uniform antes, action order and legality,
raise/call contributions, all-in flags, board/card uniqueness, refunds,
main/side pots, payouts, rake, and final conservation. Unsupported state
actions or disagreements return a contextual `ReplayError`.

Known contender cards allow exact ranking and odd-chip share verification.
If a contested showdown omits an eligible player's cards, the replay still
checks eligibility, recorded payments, rake, and conservation, and reports
`verification.winners_by_cards == false`. It does not invent missing cards.

`ReplayedHand::transitions()` borrows recorded `ReplayStep`s containing the
original OHH action number/player ID, the observed `AgentAction`, and before/
after `GameState`s. The current representation clones states per observed
decision; it does not include forced bets or dealing as agent decisions.
Compact snapshots or deltas remain future performance work.

## Remaining floating-point audit

The source audit found 437 float/epsilon/comparison-related occurrences in
55 files, including documentation and tests. The following categories explain
the monetary-looking remnants; they are not cash-ledger storage.

| Location/category | Why it remains floating point |
| --- | --- |
| CFR generators; comparison/CLI stack ranges in big blinds | Sizing multipliers and strategy probabilities; `apply_ratio` produces integer sizes |
| CFR mapper | Local logarithms/interpolation only; amounts, special-action equality, endpoints, and final bounds are integers |
| CFR engine, nodes, state, historian, fast-forward averages, budgets | Utilities, regrets, strategy weights, means, and convergence thresholds after exact monetary outcomes |
| Random agents, hand estimator, preflop charts, Monte Carlo, outs | Probabilities, equity, and mixed-strategy weights |
| Stats/comparison and `HoldemCompetition` | Means, percentages, ROI, and big-blind-normalized results derived from integer raw profit/investment |
| CLI diagnostics and TUI | Statistical plots, normalized profit, progress, geometry, colors, and display metrics |
| Core statistical tests, CFR diagnostics, benchmarks, fuzz probability validation | Chi-square/variance checks and genuine utility/probability data |
| Independent `simulated_icm` | Expected tournament payouts, computed from its existing integer trial inputs |
| Independent `rsp icm simulate` CLI | Legacy `Vec<f32>` chip-stack/prize inputs and centi-unit truncation; outside the NLHE cash path, retained as explicit technical debt |

The remaining `EARLY_EXIT_EPSILON` is a strategy-convergence tolerance.
Remaining floating `partial_cmp` operations sort statistics or equity.
`competition::tournament` still uses `partial_cmp` on integer starting stacks;
that comparison is exact and has no float tolerance. No `sum::<f32>()` or
floating epsilon remains in the cash pot/legality/OHH monetary path.

## Validation and known limits

Money, rake, pot conservation, CFR rewards/action indexing, OHH compatibility,
room parsing, and replay have targeted tests in their modules. The new audit
tests cover large mapper bounds, overflowing effective-range doubling, and
overflowing parser summary winnings. The accompanying validation report
distinguishes previously reported results from checks run in this workspace.
The full-suite reruns and these new Rust tests remain unexecuted here because
the workspace has no Rust toolchain.

Other technical limits remain explicit: callers can mutate public state
fields and violate invariants; pot planning asserts on inconsistent state;
cross-hand raw statistics still have finite `i64` accumulation limits;
full-state replay snapshots consume memory; unverified room policies and
unsupported monetary adjustments require additional work rather than an
inferred policy. This is not a claim that all definition-of-done checks passed.
