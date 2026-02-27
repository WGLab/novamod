# codex_notes.md — novamod working agreements

This file is a living checklist of **corrections, reminders, and “gotchas”** for working in this repo.
If you (Codex) make a mistake and we correct it, **append a new bullet under “New lessons (most recent first)”**
with: (1) what went wrong, (2) what to do next time, (3) where it lives (file/function).

Repo orientation (codex/fast):
- Training entrypoint is config-driven; see `training/` utilities and scripts. (Repo layout in README)  
- Core streaming datasets live in `training/dataset_utils.py`.  
- SignalBAM parsing + reference preprocessing happens via `bam_utils` (`bam_utils.pre_process`, `bam_utils.get_read_info`, `bam_utils.Read`).

---

## Lessons (most recent first)
- If changing(adding/removing) configurations, remember to also update json configs/docs files accordingly.
- `bam.fetch(until_eof=True)` means iteration works even for unsorted BAMs, but coordinate-range fetches require indexing.
- Be defensive: missing tags / malformed reads can happen; current code skips reads on exceptions.
- Avoid expensive per-read Python work in inner loops unless necessary (profiling first).
- (Add newest items here; keep them short and concrete.)

---

## What Codex should do before committing a change (pre-flight checklist)
1. Confirm the change is consistent with `dataset_utils.py` invariants.
2. Make sure the entire workflow is compatible with different k-mer length (k=9-21).
3. If behavior changes, update:
   - relevant config templates
   - README snippet(s) if user-facing
4. If you learned something new from a fix, append it under “New lessons”.
