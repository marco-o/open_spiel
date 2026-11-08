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

import {act} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import type {Frame, Presentation} from '../../contract/contract.ts';
import {RENDERERS} from '../../renderers/registry.ts';
import {frame} from '../../testing/frames.ts';
import {type Rendered, button, only, render} from '../../testing/render.tsx';
import {SchnapsRenderer} from './schnaps_renderer.tsx';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

const HIDDEN = [null, null, null, null, null];

// Seat 0 to lead in a Normal game, hearts trump, as the adapter writes it.
function playView(seat0: ReadonlyArray<string | null>): unknown {
  return {
    seats: [
      {seat: 0, team: 0, cards: seat0},
      {seat: 1, team: 1, cards: HIDDEN},
      {seat: 2, team: 0, cards: HIDDEN},
      {seat: 3, team: 1, cards: HIDDEN},
    ],
    phase: 'play',
    trump: 'h',
    contract: {name: 'Normal', declarer: 0, flecks: 1},
    trick: [],
    lastTrick: null,
    score: [0, 0],
    tricks: [0, 0],
    toMove: 0,
  };
}

function presentation(view: unknown): Presentation {
  return {
    renderer: 'schnaps',
    view,
    actions: [
      {id: 0, hint: {card: 'Jh'}},
      {id: 20, hint: {declare: 'Kh', label: 'Declare 40, lead Herz König'}},
    ],
  };
}

function seatFrame(seat: number, shown: Presentation): Frame {
  return frame({
    viewer: {kind: 'seat', seat},
    omniscient: false,
    presentation: shown,
  });
}

describe('SchnapsRenderer', () => {
  it('is registered as a board', () => {
    expect(RENDERERS.get('schnaps')).toBe(SchnapsRenderer);
  });

  it('plays a card and a declaration from the table', () => {
    const shown = presentation(playView(['Jh', 'Qh', 'Kh', 'Th', 'Ah']));
    const play = vi.fn();
    rendered = render(
      <SchnapsRenderer
        frame={seatFrame(0, shown)}
        presentation={shown}
        play={play}
      />,
    );
    const {container} = rendered;
    expect(container.textContent).toContain('Player 0 to play');
    expect(container.textContent).toContain('♥ Herz');
    expect(container.textContent).toContain('Normal (Player 0) ×2');
    const unter = only(container, 'button[aria-label="Herz Unter: play"]');
    act(() => {
      unter.click();
    });
    expect(play).toHaveBeenCalledWith(0);
    act(() => {
      button('Declare 40, lead Herz König', container).click();
    });
    expect(play).toHaveBeenCalledWith(20);
  });

  it('draws the viewer at the bottom and its partner across', () => {
    const shown = presentation(playView(HIDDEN));
    rendered = render(
      <SchnapsRenderer
        frame={seatFrame(2, shown)}
        presentation={shown}
        play={null}
      />,
    );
    const {container} = rendered;
    const placeOf = (label: string): string | null =>
      only(container, `section[aria-label^="${label}"]`)
        .closest('.schnaps-seat')
        ?.getAttribute('data-place') ?? null;
    expect(placeOf('Player 2')).toBe('bottom');
    expect(placeOf('Player 3')).toBe('left');
    expect(placeOf('Player 0')).toBe('top');
    expect(placeOf('Player 1')).toBe('right');
    expect(
      container.querySelectorAll('button.playing-card--playable'),
    ).toHaveLength(0);
  });

  it('shows the last trick and the result once the deal is over', () => {
    const view = {
      ...(playView([]) as Record<string, unknown>),
      phase: 'terminal',
      toMove: null,
      lastTrick: {
        cards: [
          {seat: 1, card: 'Ac'},
          {seat: 2, card: 'Tc'},
          {seat: 3, card: 'Jc'},
          {seat: 0, card: 'Qc'},
        ],
        winner: 1,
      },
    };
    const shown = presentation(view);
    rendered = render(
      <SchnapsRenderer
        frame={frame({presentation: shown, returns: [-6, 6, -6, 6]})}
        presentation={shown}
        play={null}
      />,
    );
    const {container} = rendered;
    expect(container.textContent).toContain('P1 + P3 win 6 points');
    expect(container.textContent).toContain('Last trick: Player 1 won');
  });
});
