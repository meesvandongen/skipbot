use std::array::from_fn;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::action::{Action, CardSource, PlayerId};
use crate::card::{
    BUILD_PILE_COUNT, Card, DISCARD_PILE_COUNT, HAND_SIZE, MAX_CARD_VALUE, full_deck,
};
use crate::error::{GameError, InvalidAction};
use crate::state::{
    BuildPileView, GameSettings, GameStateView, GameStatus, PlayerPublicState, TurnPhase,
};

const DEFAULT_SEED: u64 = 0x5EED_5EED_5EED_5EED;

/// Configuration required to bootstrap a game instance.
#[derive(Clone, Copy, Debug)]
pub struct GameConfig {
    pub num_players: usize,
    pub seed: u64,
    pub stock_size: Option<usize>,
}

impl GameConfig {
    pub fn new(num_players: usize, seed: u64) -> Result<Self, GameError> {
        GameSettings::new(num_players)?;
        Ok(Self { num_players, seed, stock_size: None })
    }
}

/// Builder that enables deterministic deck injection for testing and RL experiments.
pub struct GameBuilder {
    config: GameConfig,
    deck: Option<Vec<Card>>,
}

impl GameBuilder {
    pub fn new(num_players: usize) -> Result<Self, GameError> {
        Ok(Self {
            config: GameConfig::new(num_players, DEFAULT_SEED)?,
            deck: None,
        })
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    pub fn with_deck(mut self, deck: Vec<Card>) -> Self {
        self.deck = Some(deck);
        self
    }

    /// Override the default stock size (per player). When not set, the standard
    /// rules apply: 30 cards for up to 4 players, otherwise 20.
    pub fn with_stock_size(mut self, stock_size: usize) -> Self {
        self.config.stock_size = Some(stock_size);
        self
    }

    pub fn build(self) -> Result<Game, GameError> {
        Game::from_builder(self)
    }
}

/// Core Skip-Bo game engine.
pub struct Game {
    settings: GameSettings,
    status: GameStatus,
    current_player: PlayerId,
    players: Vec<PlayerState>,
    build_piles: [BuildPile; BUILD_PILE_COUNT],
    draw_pile: Vec<Card>,
    recycle_pile: Vec<Card>,
    turn_phase: TurnPhase,
    rng: StdRng,
}

impl Game {
    pub fn builder(num_players: usize) -> Result<GameBuilder, GameError> {
        GameBuilder::new(num_players)
    }

    pub fn new(config: GameConfig) -> Result<Self, GameError> {
        GameBuilder { config, deck: None }.build()
    }

    pub fn status(&self) -> GameStatus {
        self.status
    }

    pub fn settings(&self) -> GameSettings {
        self.settings
    }

    pub fn current_player(&self) -> PlayerId {
        self.current_player
    }

    pub fn turn_phase(&self) -> TurnPhase {
        self.turn_phase
    }

    pub fn state_view(&self, perspective: PlayerId) -> Result<GameStateView, GameError> {
        if perspective >= self.players.len() {
            return Err(GameError::InvalidPlayer(perspective));
        }
        let build_piles = from_fn(|idx| self.build_piles[idx].as_view());
        let players = self
            .players
            .iter()
            .enumerate()
            .map(|(idx, player)| PlayerPublicState {
                id: idx,
                stock_count: player.stock.len(),
                stock_top: player.stock.last().copied(),
                discard_tops: player.discard_tops(),
                discard_counts: player.discard_counts(),
                hand_size: player.hand.len(),
                is_current: idx == self.current_player,
                has_won: player.has_won,
            })
            .collect();

        Ok(GameStateView {
            settings: self.settings,
            phase: self.turn_phase,
            status: self.status,
            self_player: perspective,
            current_player: self.current_player,
            draw_pile_count: self.draw_pile.len(),
            recycle_pile_count: self.recycle_pile.len(),
            build_piles,
            players,
            hand: self.players[perspective].hand.clone(),
        })
    }

    pub fn legal_actions(&self, player: PlayerId) -> Result<Vec<Action>, GameError> {
        if matches!(self.status, GameStatus::Finished { .. }) {
            return Ok(Vec::new());
        }
        if player >= self.players.len() {
            return Err(GameError::InvalidPlayer(player));
        }
        if player != self.current_player {
            return Err(GameError::NotPlayersTurn);
        }
        let player_state = &self.players[player];
        let mut actions = Vec::new();
        let required_values: [u8; BUILD_PILE_COUNT] =
            from_fn(|idx| self.build_piles[idx].next_value());

        for (hand_index, card) in player_state.hand.iter().enumerate() {
            for (build_index, required) in required_values.iter().enumerate() {
                if card.matches_value(*required) {
                    actions.push(Action::Play {
                        source: CardSource::Hand(hand_index),
                        build_pile: build_index,
                    });
                }
            }
        }

        if let Some(card) = player_state.stock.last() {
            for (build_index, required) in required_values.iter().enumerate() {
                if card.matches_value(*required) {
                    actions.push(Action::Play {
                        source: CardSource::Stock,
                        build_pile: build_index,
                    });
                }
            }
        }

        for discard_index in 0..DISCARD_PILE_COUNT {
            if let Some(card) = player_state.discard_top(discard_index) {
                for (build_index, required) in required_values.iter().enumerate() {
                    if card.matches_value(*required) {
                        actions.push(Action::Play {
                            source: CardSource::Discard(discard_index),
                            build_pile: build_index,
                        });
                    }
                }
            }
        }

        if !player_state.hand.is_empty() {
            for discard_index in 0..DISCARD_PILE_COUNT {
                for hand_index in 0..player_state.hand.len() {
                    actions.push(Action::Discard {
                        hand_index,
                        discard_pile: discard_index,
                    });
                }
            }
        } else {
            actions.push(Action::EndTurn);
        }

        Ok(actions)
    }

    pub fn apply_action(&mut self, player: PlayerId, action: Action) -> Result<(), GameError> {
        if matches!(self.status, GameStatus::Finished { .. }) {
            return Err(GameError::GameOver);
        }
        if player >= self.players.len() {
            return Err(GameError::InvalidPlayer(player));
        }
        if player != self.current_player {
            return Err(GameError::NotPlayersTurn);
        }

        match action {
            Action::Play { source, build_pile } => self.play_card(build_pile, source)?,
            Action::Discard {
                hand_index,
                discard_pile,
            } => {
                self.discard_card(hand_index, discard_pile)?;
                self.advance_turn();
            }
            Action::EndTurn => {
                if !self.players[player].hand.is_empty() {
                    return Err(InvalidAction::MustDiscard.into());
                }
                self.advance_turn();
            }
        }

        Ok(())
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.status, GameStatus::Finished { .. })
    }

    pub fn winner(&self) -> Option<PlayerId> {
        match self.status {
            GameStatus::Finished { winner } => Some(winner),
            _ => None,
        }
    }

    fn from_builder(builder: GameBuilder) -> Result<Self, GameError> {
        let GameBuilder { config, deck } = builder;
        let mut settings = GameSettings::new(config.num_players)?;
        if let Some(custom_stock) = config.stock_size {
            if custom_stock == 0 {
                return Err(GameError::InvalidConfiguration("stock size must be positive"));
            }
            settings.stock_size = custom_stock;
        }
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut deck = if let Some(deck) = deck {
            deck
        } else {
            let mut deck = full_deck();
            deck.shuffle(&mut rng);
            deck
        };

        let required_stock_cards = settings.stock_size * settings.num_players;
        if deck.len() < required_stock_cards {
            return Err(GameError::InvalidConfiguration(
                "deck does not contain enough cards to deal stocks",
            ));
        }

        let mut players = Vec::with_capacity(settings.num_players);
        for _ in 0..settings.num_players {
            let mut stock = Vec::with_capacity(settings.stock_size);
            for _ in 0..settings.stock_size {
                stock.push(deck.pop().ok_or(GameError::InvalidConfiguration(
                    "deck exhausted while dealing stocks",
                ))?);
            }
            players.push(PlayerState::new(stock));
        }

        let mut game = Game {
            settings,
            status: GameStatus::Ongoing,
            current_player: 0,
            players,
            build_piles: from_fn(|_| BuildPile::new()),
            draw_pile: deck,
            recycle_pile: Vec::new(),
            turn_phase: TurnPhase::AwaitingAction,
            rng,
        };

        game.begin_turn();
        Ok(game)
    }

    fn begin_turn(&mut self) {
        if matches!(self.status, GameStatus::Finished { .. }) {
            self.turn_phase = TurnPhase::GameOver;
            return;
        }
        self.turn_phase = TurnPhase::AwaitingAction;
        let current = self.current_player;
        let hand_target = self.settings.hand_size;
        while self.players[current].hand.len() < hand_target {
            match self.draw_card() {
                Some(card) => self.players[current].hand.push(card),
                None => break,
            }
        }
    }

    fn advance_turn(&mut self) {
        if matches!(self.status, GameStatus::Finished { .. }) {
            self.turn_phase = TurnPhase::GameOver;
            return;
        }
        self.current_player = (self.current_player + 1) % self.players.len();
        self.begin_turn();
    }

    fn play_card(&mut self, build_pile_idx: usize, source: CardSource) -> Result<(), GameError> {
        if build_pile_idx >= BUILD_PILE_COUNT {
            return Err(InvalidAction::BuildPileIndex(build_pile_idx).into());
        }
        if matches!(self.turn_phase, TurnPhase::GameOver) {
            return Err(GameError::GameOver);
        }
        let required_value = self.build_piles[build_pile_idx].next_value();
        {
            let player_state = &self.players[self.current_player];
            let card = Self::peek_source(player_state, source)?;
            if !card.matches_value(required_value) {
                return Err(InvalidAction::CardMismatch {
                    required: required_value,
                }
                .into());
            }
        }
        let card = {
            let player_state = &mut self.players[self.current_player];
            Self::take_from_source(player_state, source)?
        };
        self.build_piles[build_pile_idx].push(card);
        if self.build_piles[build_pile_idx].is_complete() {
            let completed = self.build_piles[build_pile_idx].take_cards();
            self.recycle_pile.extend(completed);
        }
        if self.players[self.current_player].stock.is_empty() {
            self.players[self.current_player].has_won = true;
            self.status = GameStatus::Finished {
                winner: self.current_player,
            };
            self.turn_phase = TurnPhase::GameOver;
        }
        Ok(())
    }

    fn discard_card(&mut self, hand_index: usize, discard_index: usize) -> Result<(), GameError> {
        if discard_index >= DISCARD_PILE_COUNT {
            return Err(InvalidAction::DiscardIndex(discard_index).into());
        }
        let player_state = &mut self.players[self.current_player];
        if hand_index >= player_state.hand.len() {
            return Err(InvalidAction::HandIndex(hand_index).into());
        }
        let card = player_state.hand.remove(hand_index);
        player_state.discard_piles[discard_index].push(card);
        Ok(())
    }

    fn draw_card(&mut self) -> Option<Card> {
        if let Some(card) = self.draw_pile.pop() {
            return Some(card);
        }
        if self.recycle_pile.is_empty() {
            return None;
        }
        self.reshuffle_recycle();
        self.draw_pile.pop()
    }

    fn reshuffle_recycle(&mut self) {
        self.recycle_pile.shuffle(&mut self.rng);
        self.draw_pile.append(&mut self.recycle_pile);
    }

    fn peek_source<'a>(player: &'a PlayerState, source: CardSource) -> Result<&'a Card, GameError> {
        match source {
            CardSource::Hand(index) => player
                .hand
                .get(index)
                .ok_or_else(|| InvalidAction::HandIndex(index).into()),
            CardSource::Stock => player
                .stock
                .last()
                .ok_or(InvalidAction::NoCardAvailable.into()),
            CardSource::Discard(index) => {
                if index >= DISCARD_PILE_COUNT {
                    Err(InvalidAction::DiscardIndex(index).into())
                } else {
                    player
                        .discard_piles
                        .get(index)
                        .and_then(|pile| pile.last())
                        .ok_or(InvalidAction::NoCardAvailable.into())
                }
            }
        }
    }

    fn take_from_source(player: &mut PlayerState, source: CardSource) -> Result<Card, GameError> {
        match source {
            CardSource::Hand(index) => {
                if index >= player.hand.len() {
                    return Err(InvalidAction::HandIndex(index).into());
                }
                Ok(player.hand.remove(index))
            }
            CardSource::Stock => player
                .stock
                .pop()
                .ok_or(InvalidAction::NoCardAvailable.into()),
            CardSource::Discard(index) => {
                if index >= DISCARD_PILE_COUNT {
                    Err(InvalidAction::DiscardIndex(index).into())
                } else {
                    player
                        .discard_piles
                        .get_mut(index)
                        .and_then(|pile| pile.pop())
                        .ok_or(InvalidAction::NoCardAvailable.into())
                }
            }
        }
    }
}

#[derive(Clone)]
struct PlayerState {
    stock: Vec<Card>,
    hand: Vec<Card>,
    discard_piles: [Vec<Card>; DISCARD_PILE_COUNT],
    has_won: bool,
}

impl PlayerState {
    fn new(mut stock: Vec<Card>) -> Self {
        // Reveal the top card (no-op in this representation because top is last).
        stock.shrink_to_fit();
        Self {
            stock,
            hand: Vec::with_capacity(HAND_SIZE),
            discard_piles: from_fn(|_| Vec::new()),
            has_won: false,
        }
    }

    fn discard_top(&self, index: usize) -> Option<Card> {
        self.discard_piles
            .get(index)
            .and_then(|pile| pile.last())
            .copied()
    }

    fn discard_tops(&self) -> [Option<Card>; DISCARD_PILE_COUNT] {
        from_fn(|idx| self.discard_top(idx))
    }

    fn discard_counts(&self) -> [usize; DISCARD_PILE_COUNT] {
        from_fn(|idx| self.discard_piles[idx].len())
    }
}

#[derive(Clone)]
struct BuildPile {
    cards: Vec<Card>,
}

impl BuildPile {
    fn new() -> Self {
        Self {
            cards: Vec::with_capacity(MAX_CARD_VALUE as usize),
        }
    }

    fn next_value(&self) -> u8 {
        (self.cards.len() as u8 % MAX_CARD_VALUE) + 1
    }

    fn push(&mut self, card: Card) {
        self.cards.push(card);
    }

    fn is_complete(&self) -> bool {
        self.cards.len() == MAX_CARD_VALUE as usize
    }

    fn take_cards(&mut self) -> Vec<Card> {
        std::mem::take(&mut self.cards)
    }

    fn as_view(&self) -> BuildPileView {
        BuildPileView {
            cards: self.cards.clone(),
            next_value: self.next_value(),
        }
    }
}
