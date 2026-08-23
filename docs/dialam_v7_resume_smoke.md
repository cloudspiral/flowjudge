# DialAM v7 interruption/resume smoke

Decision: **PASS**.

The n=256 run intentionally terminated only after committing the full
step-8 Trainer state to the Modal volume. The required-resume invocation
continued with step 9 and finished at step 32; the final adapter reloaded
successfully. A third auto-mode invocation returned the identical adapter
hash and completed manifest without executing another training step.

This proves recovery of model/adapter, optimizer, scheduler, RNG, Trainer
state, and sample position after the first periodic checkpoint. The full
v7 run uses the same mechanism with a 100-step interval and retains the
two newest full states.

Raw run results and checkpoints remain local/ignored and on the private
Modal volume. Their paths and hashes are recorded in
`reports/dialam_v7_resume_smoke.json`.
