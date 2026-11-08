# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Replaced by the schnaps ObservationStruct when available.
"""The schnaps (Bauernschnapsen) adapter: a four-seat card table.

The view is the record the `schnaps` client renderer reads:

- `seats`: one entry per seat, in seat order: `seat`, `team` (0 for seats 0
  and 2, 1 for seats 1 and 3) and `cards`, the hand as card codes, or one
  `None` per card the viewer may not see.
- `phase`: "deal", "trump", "bidding", "play" or "terminal".
- `trump`: the trump suit as a client suit code ("h", "d", "s", "c"), or
  None before it is chosen.
- `contract`: `{"name", "declarer", "flecks"}` once bidding starts, else None.
- `trick`: the cards of the trick in play, each `{"seat", "card"}`.
- `lastTrick`: `{"cards", "winner"}` for the last finished trick, or None.
- `score` and `tricks`: the card points and tricks of each team.
- `toMove`: the seat to act, or None.

Cards use the client's codes: the suits Herz, Caro, Pik and Kreuz are drawn
as hearts, diamonds, spades and clubs ("h", "d", "s", "c"), and the ranks
Unter, Ober, Koenig, Zehn and As as "J", "Q", "K", "T" and "A"; the
renderer's skin labels them U, O, K, 10 and A.

Each legal action is hinted for the acting seat: a card played is
`{"card": code}`, a 20/40 declaration is `{"declare": code, "label": text}`
(the card it leads), and every other action is `{"button": text}`.

The game has no Python bindings beyond `pyspiel.State`, so the adapter reads
the state's text (`ToString`) and its history.
"""

from collections.abc import Mapping, Sequence
import re

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

GAME = "schnaps"

Json = contract.JsonValue

_NUM_SEATS = 4
_NUM_RANKS = 5
_NUM_CARDS = 20
_HAND_SIZE = 5

# The game's numbering: card = suit * 5 + rank.
_GAME_RANKS = ("U", "O", "K", "Z", "A")
_GAME_SUITS = ("h", "c", "p", "k")  # Herz, Caro, Pik, Kreuz
_CLIENT_RANKS = ("J", "Q", "K", "T", "A")
_CLIENT_SUITS = ("h", "d", "s", "c")
_SUIT_NAMES = ("Herz", "Caro", "Pik", "Kreuz")
_RANK_NAMES = ("Unter", "Ober", "König", "Zehn", "As")

_FIRST_DECLARE = 20
_FIRST_TRUMP = 28
_PASS = 32
_FLECK = 33
_FIRST_CALL = 34
_CALLS = ("Schnapser", "Lomt", "Bauern", "Jodler", "Hermort")

_PHASES = {
    "DealCaller": "deal",
    "ChooseTrump": "trump",
    "DealRest": "deal",
    "Bidding": "bidding",
    "Play": "play",
    "Terminal": "terminal",
}

_CONTRACT = re.compile(r"^Contract: (\w+) by P(\d), flecks=(\d+)$")
_SCORE = re.compile(r"^Score: (\d+)-(\d+)\s+Tricks: (\d+)-(\d+)$")
_TRICK_PLAY = re.compile(r"^P(\d):(\w\w)$")


def card_code(card: int) -> str:
  """The client's code of the card numbered `card`: "Kh" for Herz Koenig."""
  if not 0 <= card < _NUM_CARDS:
    raise ValueError(f"schnaps: {card} is not a card.")
  return _CLIENT_RANKS[card % _NUM_RANKS] + _CLIENT_SUITS[card // _NUM_RANKS]


def card_name(card: int) -> str:
  """The card's German name: "Herz König"."""
  return f"{_SUIT_NAMES[card // _NUM_RANKS]} {_RANK_NAMES[card % _NUM_RANKS]}"


def _parse_game_card(text: str) -> int:
  """The card numbered by the game's text: "Kh" is Herz Koenig."""
  if len(text) != 2:
    raise ValueError(f"schnaps: {text!r} is not a card.")
  rank = _GAME_RANKS.index(text[0])
  suit = _GAME_SUITS.index(text[1])
  return suit * _NUM_RANKS + rank


def _declared_card(action: int) -> int:
  """The card a 20/40 declaration leads: its suit's Koenig or Ober."""
  index = action - _FIRST_DECLARE
  return (index // 2) * _NUM_RANKS + (2 if index % 2 == 0 else 1)


class _Parsed:
  """The state's text, read into fields."""

  def __init__(self, text: str):
    self.phase = ""
    self.trump: int | None = None
    self.hands: list[list[int]] = [[] for _ in range(_NUM_SEATS)]
    self.contract: tuple[str, int, int] | None = None
    self.trick: list[tuple[int, int]] = []
    self.score = [0, 0]
    self.tricks = [0, 0]
    for line in text.splitlines():
      line = line.rstrip()
      if line.startswith("Phase: "):
        self.phase = line[len("Phase: "):]
      elif line.startswith("Trump: "):
        self.trump = _GAME_SUITS.index(line[len("Trump: "):])
      elif re.match(r"^P\d:", line):
        seat = int(line[1])
        cards = line[3:].split()
        self.hands[seat] = [_parse_game_card(card) for card in cards]
      elif line.startswith("Contract: "):
        match = _CONTRACT.match(line)
        if match is None:
          raise ValueError(f"schnaps: cannot read {line!r}.")
        self.contract = (match[1], int(match[2]), int(match[3]))
      elif line.startswith("Trick:"):
        for play in line[len("Trick:"):].split():
          match = _TRICK_PLAY.match(play)
          if match is None:
            raise ValueError(f"schnaps: cannot read the trick {line!r}.")
          self.trick.append((int(match[1]), _parse_game_card(match[2])))
      elif line.startswith("Score: "):
        match = _SCORE.match(line)
        if match is None:
          raise ValueError(f"schnaps: cannot read {line!r}.")
        self.score = [int(match[1]), int(match[2])]
        self.tricks = [int(match[3]), int(match[4])]
    if self.phase not in _PHASES:
      raise ValueError(f"schnaps: unknown phase {self.phase!r}.")


def _plays(state: pyspiel.State) -> list[tuple[int, int]]:
  """Every card played so far, in order, as (seat, card).

  A declaration leads its Koenig or Ober, so it is a card played too.
  """
  plays = []
  for item in state.full_history():
    if item.player < 0:
      continue
    if item.action < _NUM_CARDS:
      plays.append((item.player, item.action))
    elif item.action < _FIRST_TRUMP:
      plays.append((item.player, _declared_card(item.action)))
  return plays


def _trick_winner(
    trick: Sequence[tuple[int, int]], trump: int | None
) -> int:
  """The seat that wins a finished trick, as the game resolves it."""

  def beats(a: int, b: int) -> bool:
    ta = trump is not None and a // _NUM_RANKS == trump
    tb = trump is not None and b // _NUM_RANKS == trump
    if ta != tb:
      return ta
    if a // _NUM_RANKS == b // _NUM_RANKS:
      return a % _NUM_RANKS > b % _NUM_RANKS
    return False

  best = 0
  for i in range(1, len(trick)):
    if beats(trick[i][1], trick[best][1]):
      best = i
  return trick[best][0]


def _last_trick(state: pyspiel.State, parsed: _Parsed) -> Json:
  """The last finished trick and its winner, or None before the first."""
  plays = _plays(state)
  finished = len(plays) // _NUM_SEATS
  if finished == 0:
    return None
  trick = plays[(finished - 1) * _NUM_SEATS : finished * _NUM_SEATS]
  # Lomt is played without trump.
  lomt = parsed.contract is not None and parsed.contract[0] == "Lomt"
  trump = None if lomt else parsed.trump
  return {
      "cards": [{"seat": seat, "card": card_code(card)} for seat, card in trick],
      "winner": _trick_winner(trick, trump),
  }


def _action_text(state: pyspiel.State, action: int) -> str:
  """A legal action, as a button or a history row names it."""
  if action < _NUM_CARDS:
    return f"Play {card_name(action)}"
  if action < _FIRST_TRUMP:
    card = _declared_card(action)
    trump = _Parsed(str(state)).trump
    points = "40" if card // _NUM_RANKS == trump else "20"
    return f"Declare {points}, lead {card_name(card)}"
  if action < _PASS:
    return f"Trump: {_SUIT_NAMES[action - _FIRST_TRUMP]}"
  if action == _PASS:
    return "Pass"
  if action == _FLECK:
    return "Fleck"
  if _FIRST_CALL <= action < _FIRST_CALL + len(_CALLS):
    return f"Call {_CALLS[action - _FIRST_CALL]}"
  raise ValueError(f"schnaps: {action} is not an action.")


class SchnapsAdapter(adapter.Adapter):
  """Presents a schnaps deal to one viewer as a four-seat card table."""

  renderer = "schnaps"

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, Json]:
    """The table, holding only the hands the viewer may see."""
    if state.get_game().get_type().short_name != GAME:
      raise ValueError("The schnaps adapter was given another game.")
    parsed = _Parsed(str(state))
    seats: list[Json] = []
    for seat in range(_NUM_SEATS):
      hand = parsed.hands[seat]
      cards: list[Json] = (
          [card_code(card) for card in hand]
          if viewpoint.sees_private(seat)
          else [None] * len(hand)
      )
      seats.append({"seat": seat, "team": seat % 2, "cards": cards})
    contract_view: Json = None
    if parsed.contract is not None:
      name, declarer, flecks = parsed.contract
      contract_view = {"name": name, "declarer": declarer, "flecks": flecks}
    player = state.current_player()
    return {
        "seats": seats,
        "phase": _PHASES[parsed.phase],
        "trump": (
            _CLIENT_SUITS[parsed.trump] if parsed.trump is not None else None
        ),
        "contract": contract_view,
        "trick": [
            {"seat": seat, "card": card_code(card)}
            for seat, card in parsed.trick
        ],
        "lastTrick": _last_trick(state, parsed),
        "score": list(parsed.score),
        "tricks": list(parsed.tricks),
        "toMove": int(player) if 0 <= player < _NUM_SEATS else None,
    }

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: adapter.Viewpoint,
  ) -> Mapping[int, Json]:
    """A card to press for each card played; a button for anything else."""
    del viewpoint  # The actions are the acting seat's, whom it may see.
    hints: dict[int, Json] = {}
    for action in actions:
      if action < _NUM_CARDS:
        hints[action] = {"card": card_code(action)}
      elif action < _FIRST_TRUMP:
        hints[action] = {
            "declare": card_code(_declared_card(action)),
            "label": _action_text(state, action),
        }
      else:
        hints[action] = {"button": _action_text(state, action)}
    return hints

  def public_action_label(
      self, state: pyspiel.State, action: int
  ) -> str | None:
    """Every call and card played is public; the deal stays hidden."""
    if state.current_player() == pyspiel.PlayerId.CHANCE:
      return None
    return _action_text(state, action)


ADAPTER = SchnapsAdapter()
