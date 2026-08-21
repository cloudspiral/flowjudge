# FlowJudge error-driven v2 results

## Diagnosed MVP failure

The v1 n=96 checkpoint was the first to learn any held-out response structure:
its normalized edge F1 rose from the 0.6B base model's 0% to 16%, and valid JSON
rose from 66.7% to 100%. However, it emitted exactly one edge for each of the
nine own-eval cases, even though the gold set contains 16 edges, and it usually
attached that edge to a nearby wrong unit. Exact graph match remained 0%.

## Data-only revisions

Training settings were held fixed at three epochs, rank-8 LoRA over 16 layers,
1e-4 learning rate, batch size 1, gradient accumulation 4, prompt masking, and
the same 4-bit Qwen3 0.6B base.

| Dataset | Data change | Valid JSON | Exact graph | Edge precision | Edge recall | Edge F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 n=96 | 96 filtered real examples | 100% | 0% | 22.2% | 12.5% | 16.0% |
| v2a n=120 | 24 ID-remapped variants of attachment-hard real examples | 100% | 0% | 11.1% | 6.3% | 8.0% |
| v2b n=120 | 24 three-edge curriculum chains with adjacent hard negatives | 100% | 0% | 22.2% | 12.5% | 16.0% |

The first revision made performance worse and was rejected. The second restored
the v1 result but did not improve it, so it also did not resolve the one-edge
ceiling. These negative results are retained rather than selecting only a
favorable run.

## Capacity check

After two data-only revisions failed to improve the ceiling, the same v2b data
and recipe were tested on the assignment-recommended Qwen3 1.7B size. Its tuned
edge F1 was 8.0%, below its own base model's 9.1%, so that checkpoint was also
rejected. The best measured local checkpoint remains Qwen3 0.6B v1 n=96.

## Honest conclusion

No tested dataset size reliably holds the full behavior. The v1 n=96 checkpoint
does prove a narrow training effect—portable bare JSON and nonzero held-out edge
reconstruction versus a zero-F1 base—but not reliable graph reconstruction and
not superiority to the weak frontier baselines. The minimum viable dataset size
is therefore **not established at N ≤ 120**. The next iteration should improve
gold-label alignment and add more diverse, human-adjudicated attachment
contrasts before spending time on training configuration.

Fixed Sol rubric scores are pending for n=48, n=96, and v2 because the account
returned `insufficient_quota`. Raw model responses and deterministic metrics are
complete; no substitute judge was introduced.
