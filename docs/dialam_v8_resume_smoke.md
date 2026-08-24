# DialAM v8 interruption/resume smoke

Decision: **PASS**.

The n=252 run intentionally terminated only after committing the full
step-8 Trainer state to the Modal volume. The required-resume invocation
executed step 9 next and finished at step 32; the final adapter reloaded
successfully. A third auto-mode invocation returned the identical adapter
hash and completed manifest without executing another training step.

This proves recovery of adapter, optimizer, scheduler, RNG, Trainer state,
and sample position for the v8 listwise objective. The full run uses the
same implementation with a 100-step save interval and retains the newest
two complete states.

Raw run results and checkpoints remain local/ignored and on the private
Modal volume. Their paths and hashes are recorded in
`reports/dialam_v8_resume_smoke.json`.
