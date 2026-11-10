# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The steps of `oot2 prepare` in order (architecture D24c) and the runner that keeps their record (D5).

Each step ticket adds its step to all_steps() when it lands: extraction and module decoding, lift, compile and link,
then the shader steps. A step names the earlier steps it reads from in Step.needs. The translate step needs only the
extraction and comes right after it, so that a translator that cannot work stops the run in minutes, not after the
compile. The ledger step reads what the game recorded on its last run and translates it as variants; it comes right
after the translate step and no later step reads its output. The pipelines step builds every pipeline library and compute
pipeline of the shader cache with the player's GPU into the application pipeline cache; it reads only the translate
step's pack and comes before the lift, so that a GPU that cannot build stops the run early and the long step is the one
a stopped run resumes first. The font step places the font the runtime serves for the console's shared font calls; it
reads no other step's output and comes right after the extraction, so that a machine that cannot fetch the font stops
the run in minutes too.
"""

from __future__ import annotations

from .check import Check
from .extract import Extract
from .font import Font
from .ledger import Ledger
from .modules import Modules
from .pipelines import Pipelines
from .program import Compile, Lift, Link
from .runner import Step
from .translate import Translate


def all_steps() -> list[Step]:
    return [Check(), Extract(), Font(), Translate(), Ledger(), Pipelines(), Modules(), Lift(), Compile(), Link()]
