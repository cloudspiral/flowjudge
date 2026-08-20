# VivesDebate source subset

These CSV files are unmodified source files from the official VivesDebate
Zenodo release:

- Dataset DOI: <https://doi.org/10.5281/zenodo.6531487> (version 3)
- Paper DOI: <https://doi.org/10.3390/app11157160>
- Authors: Ramon Ruiz-Dolz, Montserrat Nofre, Mariona Taulé, Stella Heras,
  and Ana García-Fornes
- Dataset license: CC BY-NC-SA 4.0

The selected English ADUs are machine translations supplied by the corpus, not
translations produced by FlowJudge. The raw Catalan and Spanish columns are
retained alongside English in both these files and the converted JSONL.

`manifest.json` records the exact release, selection rule, retrieval date, and
official MD5 checksums. Debates 1–10 are selected. Every valid RA/CA/MA target is
converted. One malformed relation slot in the version 3 files is retained as a
structured source issue and never imputed.

The VivesDebate license applies to these source files and the converted benchmark
data derived from them. Project code and original synthetic cases are separate.
