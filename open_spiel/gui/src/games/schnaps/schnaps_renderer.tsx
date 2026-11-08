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

// The schnaps renderer: a four-seat table laid out as a bridge table, the
// bottom seat's partner across, play running clockwise (bottom, left, top,
// right), the trick in the middle and the contract and score in a corner.

import type {JSX} from 'react';

import {CardFace} from '../../cards/card_face.tsx';
import {cardActions} from '../../cards/card_hints.ts';
import {Hand} from '../../cards/hand.tsx';
import type {Frame} from '../../contract/contract.ts';
import {sessionModeOf, useBottomSeat} from '../../game/bottom_seat.ts';
import {readHints} from '../../renderers/hints.ts';
import type {RendererEntry, RendererProps} from '../../renderers/renderer.ts';
import {field, record, string} from '../../renderers/struct_reader.ts';
import {
  SCHNAPS_SKIN,
  type SchnapsPlay,
  type SchnapsView,
  readSchnapsView,
  suitGlyph,
  suitName,
} from './schnaps_view.ts';

import './schnaps.css';

/** Where a seat sits, relative to the seat drawn at the bottom. */
type Place = 'bottom' | 'left' | 'top' | 'right';

const PLACES: readonly Place[] = ['bottom', 'left', 'top', 'right'];

function placeOf(seat: number, bottom: number): Place {
  return PLACES[(seat - bottom + 4) % 4] ?? 'bottom';
}

function seatName(seat: number): string {
  return `Player ${String(seat)}`;
}

function teamName(team: number): string {
  return team === 0 ? 'P0 + P2' : 'P1 + P3';
}

function statusText(view: SchnapsView, frame: Frame): string {
  const mover = view.toMove === null ? '' : seatName(view.toMove);
  switch (view.phase) {
    case 'deal':
      return 'Dealing…';
    case 'trump':
      return `${mover} chooses trump from the first three cards`;
    case 'bidding':
      return `Bidding: ${mover} calls, flecks or passes`;
    case 'play':
      return `${mover} to play`;
    case 'terminal': {
      const returns = frame.returns ?? [];
      const value = returns[0] ?? 0;
      const winner = value >= 0 ? 0 : 1;
      return `${teamName(winner)} win ${String(Math.abs(value))} points`;
    }
    default: {
      const unreachable: never = view.phase;
      throw new Error(`Unhandled phase: ${String(unreachable)}`);
    }
  }
}

/** The trump, the contract and each team's points and tricks. */
function InfoBox({view}: {readonly view: SchnapsView}): JSX.Element {
  const {trump, contract} = view;
  return (
    <dl className="schnaps-info" aria-label="Deal">
      <div className="schnaps-info__item">
        <dt>Trump</dt>
        <dd>
          {trump === null ? '—' : `${suitGlyph(trump)} ${suitName(trump)}`}
        </dd>
      </div>
      <div className="schnaps-info__item">
        <dt>Contract</dt>
        <dd>
          {contract === null
            ? '—'
            : `${contract.name} (${seatName(contract.declarer)})`}
          {contract !== null && contract.flecks > 0
            ? ` ×${String(2 ** contract.flecks)}`
            : ''}
        </dd>
      </div>
      {[0, 1].map(team => (
        <div className="schnaps-info__item" key={team}>
          <dt>{teamName(team)}</dt>
          <dd>
            {`${String(view.score[team] ?? 0)} pts · ${String(view.tricks[team] ?? 0)} tricks`}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** The trick in play, or the last one, each card before its player. */
function TrickArea({
  view,
  bottom,
}: {
  readonly view: SchnapsView;
  readonly bottom: number;
}): JSX.Element {
  const last = view.trick.length === 0 ? view.lastTrick : null;
  const showingLast = last !== null;
  const plays: readonly SchnapsPlay[] = last === null ? view.trick : last.cards;
  const caption =
    last === null ? 'Trick' : `Last trick: ${seatName(last.winner)} won`;
  return (
    <section
      className="schnaps-trick"
      aria-label={caption}
      data-last={showingLast ? 'true' : undefined}
    >
      {plays.map(play => (
        <div
          className="schnaps-trick__card"
          data-place={placeOf(play.seat, bottom)}
          key={play.seat}
        >
          <CardFace code={play.card} skin={SCHNAPS_SKIN} />
        </div>
      ))}
      <p className="schnaps-trick__caption">{caption}</p>
    </section>
  );
}

/** One seat's hand at its place, marked while it is that seat's turn. */
function SeatArea({
  view,
  seat,
  bottom,
  playable,
  play,
}: {
  readonly view: SchnapsView;
  readonly seat: number;
  readonly bottom: number;
  readonly playable: ReadonlyMap<string, number>;
  readonly play: ((action: number) => void) | null;
}): JSX.Element {
  const entry = view.seats[seat];
  const roles = [
    seat === 0 ? 'caller' : null,
    view.contract !== null &&
    view.contract.name !== 'Normal' &&
    view.contract.declarer === seat
      ? 'declarer'
      : null,
  ].filter(role => role !== null);
  const label =
    roles.length === 0
      ? seatName(seat)
      : `${seatName(seat)} (${roles.join(', ')})`;
  return (
    <div
      className="schnaps-seat"
      data-place={placeOf(seat, bottom)}
      data-active={view.toMove === seat ? 'true' : undefined}
    >
      <Hand
        label={label}
        cards={entry?.cards ?? []}
        playable={playable}
        play={play}
        skin={SCHNAPS_SKIN}
        fanned={seat !== bottom}
      />
    </div>
  );
}

/** The buttons for every action that is not a card: calls, declarations. */
function MoveButtons({
  presentation,
  play,
}: Pick<RendererProps, 'presentation' | 'play'>): JSX.Element | null {
  const buttons = readHints(presentation, (hint, path) => {
    if (hint === null) return null;
    const fields = record(hint, path);
    if ('button' in fields) return field(hint, path, 'button', string);
    if ('declare' in fields) return field(hint, path, 'label', string);
    return null;
  });
  if (buttons.length === 0) return null;
  return (
    <div className="schnaps-moves" role="group" aria-label="Moves">
      {buttons.map(({action, hint: text}) => (
        <button
          type="button"
          className="button"
          key={action}
          disabled={play === null}
          onClick={() => {
            play?.(action);
          }}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

/** Draws a schnaps frame as a bridge-style card table. */
export function SchnapsRenderer({
  frame,
  presentation,
  play,
}: RendererProps): JSX.Element {
  const view = readSchnapsView(presentation.view);
  const {seat: viewerSeat} = frame.viewer;
  const {seat: bottom} = useBottomSeat(
    {mode: sessionModeOf(frame.viewer)},
    {
      seat: viewerSeat,
      viewSeat: viewerSeat,
      isParticipant: viewerSeat !== null,
    },
  );
  const playable = cardActions(presentation);
  return (
    <div className="schnaps">
      <p className="schnaps-status" role="status">
        {statusText(view, frame)}
      </p>
      <div className="schnaps-table">
        <InfoBox view={view} />
        {[0, 1, 2, 3].map(seat => (
          <SeatArea
            key={seat}
            view={view}
            seat={seat}
            bottom={bottom}
            playable={playable}
            play={play}
          />
        ))}
        <TrickArea view={view} bottom={bottom} />
      </div>
      <MoveButtons presentation={presentation} play={play} />
    </div>
  );
}

/** The schnaps renderer's registry entry. */
export const SCHNAPS_RENDERER: RendererEntry = {
  name: 'schnaps',
  render: SchnapsRenderer,
};
