# Released result data

Each dataset directory contains the per-case predictions and metrics used by
the manuscript and interactive gallery. Files named `radiancereg_gs.json`
store the final proposed-method estimates; the remaining files store outputs
from public comparison methods executed under the common protocol. Separate
timing files retain measured registration time without duplicating candidate
diagnostics.

`provenance.json` records the evaluation scope, case count, and local directory
for each released group. The release covers 24 ScanNet-GSReg cases, 13
Objaverse cases, and 19 ETH3D cases; every included method has one record for
every case in the corresponding group. Only derived predictions and metrics
are distributed. Raw datasets, third-party source code, and pretrained
comparison weights remain under their respective licenses and are not
included.
