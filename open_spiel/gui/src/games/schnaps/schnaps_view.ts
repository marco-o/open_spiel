// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The schnaps view as the server's adapter sends it (schnaps_adapter.py),
// read with the checked struct readers, and the skin its cards are drawn
// with: Unter, Ober, Koenig, Zehn and As travel as J, Q, K, T and A.

import type {Rank, Suit} from '../../cards/card.ts';
import type {Skin} from '../../cards/skins.ts';
import {
  array,
  field,
  integer,
  nullable,
  string,
} from '../../renderers/struct_reader.ts';

/** One card on the table and the seat that played it. */
export interface SchnapsPlay {
  readonly seat: number;
  readonly card: string;
}

/** One seat: its team and its hand, null for each hidden card. */
export interface SchnapsSeat {
  readonly seat: number;
  readonly team: number;
  readonly cards: ReadonlyArray<string | null>;
}

/** The contract being played. */
export interface SchnapsContract {
  readonly name: string;
  readonly declarer: number;
  readonly flecks: number;
}

/** The stage of the deal. */
export type SchnapsPhase = 'deal' | 'trump' | 'bidding' | 'play' | 'terminal';

/** The whole table, as one viewer may see it. */
export interface SchnapsView {
  readonly seats: readonly SchnapsSeat[];
  readonly phase: SchnapsPhase;
  readonly trump: Suit | null;
  readonly contract: SchnapsContract | null;
  readonly trick: readonly SchnapsPlay[];
  readonly lastTrick: {
    readonly cards: readonly SchnapsPlay[];
    readonly winner: number;
  } | null;
  readonly score: readonly number[];
  readonly tricks: readonly number[];
  readonly toMove: number | null;
}

const PHASES: readonly SchnapsPhase[] = [
  'deal',
  'trump',
  'bidding',
  'play',
  'terminal',
];
const SUITS: readonly Suit[] = ['h', 'd', 's', 'c'];

function readPhase(value: unknown, path: string): SchnapsPhase {
  const text = string(value, path);
  const phase = PHASES.find(candidate => candidate === text);
  if (phase === undefined) throw new Error(`${path}: unknown phase ${text}.`);
  return phase;
}

function readSuit(value: unknown, path: string): Suit {
  const text = string(value, path);
  const suit = SUITS.find(candidate => candidate === text);
  if (suit === undefined) throw new Error(`${path}: unknown suit ${text}.`);
  return suit;
}

function readPlays(value: unknown, path: string): SchnapsPlay[] {
  return array(value, path, (play, playPath) => ({
    seat: field(play, playPath, 'seat', integer),
    card: field(play, playPath, 'card', string),
  }));
}

function readSeat(value: unknown, path: string): SchnapsSeat {
  return {
    seat: field(value, path, 'seat', integer),
    team: field(value, path, 'team', integer),
    cards: field(value, path, 'cards', (cards, cardsPath) =>
      array(cards, cardsPath, (card, cardPath) =>
        nullable(card, cardPath, string),
      ),
    ),
  };
}

function readContract(value: unknown, path: string): SchnapsContract {
  return {
    name: field(value, path, 'name', string),
    declarer: field(value, path, 'declarer', integer),
    flecks: field(value, path, 'flecks', integer),
  };
}

function readIntegers(value: unknown, path: string): number[] {
  return array(value, path, integer);
}

/** The schnaps view of a presentation; throws naming a malformed field. */
export function readSchnapsView(view: unknown): SchnapsView {
  const path = 'view';
  return {
    seats: field(view, path, 'seats', (value, seatsPath) =>
      array(value, seatsPath, readSeat),
    ),
    phase: field(view, path, 'phase', readPhase),
    trump: field(view, path, 'trump', (value, trumpPath) =>
      nullable(value, trumpPath, readSuit),
    ),
    contract: field(view, path, 'contract', (value, contractPath) =>
      nullable(value, contractPath, readContract),
    ),
    trick: field(view, path, 'trick', readPlays),
    lastTrick: field(view, path, 'lastTrick', (value, lastPath) =>
      nullable(value, lastPath, (last, p) => ({
        cards: field(last, p, 'cards', readPlays),
        winner: field(last, p, 'winner', integer),
      })),
    ),
    score: field(view, path, 'score', readIntegers),
    tricks: field(view, path, 'tricks', readIntegers),
    toMove: field(view, path, 'toMove', (value, movePath) =>
      nullable(value, movePath, integer),
    ),
  };
}

const SUIT_NAMES: Record<Suit, string> = {
  h: 'Herz',
  d: 'Caro',
  s: 'Pik',
  c: 'Kreuz',
};

const GLYPHS: Record<Suit, string> = {h: '♥', d: '♦', s: '♠', c: '♣'};

const RANK_LABELS: Partial<Record<Rank, string>> = {
  J: 'U',
  Q: 'O',
  K: 'K',
  T: '10',
  A: 'A',
};

const RANK_NAMES: Partial<Record<Rank, string>> = {
  J: 'Unter',
  Q: 'Ober',
  K: 'König',
  T: 'Zehn',
  A: 'As',
};

/** The German name of a suit: "Herz". */
export function suitName(suit: Suit): string {
  return SUIT_NAMES[suit];
}

/** The glyph of a suit: "♥". */
export function suitGlyph(suit: Suit): string {
  return GLYPHS[suit];
}

/** The schnaps skin: U, O, K, 10 and A, with German card names. */
export const SCHNAPS_SKIN: Skin = schnapsSkin();

function schnapsSkin(): Skin {
  return {
    name: 'schnaps',
    rankLabel: rank => RANK_LABELS[rank] ?? rank,
    suitGlyph: suit => GLYPHS[suit],
    suitTone: suit => (suit === 'h' || suit === 'd' ? 'red' : 'black'),
    cardName: ({rank, suit}) =>
      `${SUIT_NAMES[suit]} ${RANK_NAMES[rank] ?? rank}`,
  };
}
