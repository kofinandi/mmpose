# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is big, but these files are relevant for your work. Read these files for full context:
   - `projects/gp_kalman_filter/GP_Kalman_Filter_Steps.md` — the GP-Kalman filter base description and steps.
   - `projects/gp_kalman_filter/filter.py` — the GP-Kalman filter implementation.
   - `mmpose/postprocessing/filters/gp_kalman_smoother.py` — the GP-Kalman filter wrapper for MM-Pose.
4. **Verify data exists**: Check that `benchmark/predictions/20260622_emdb_topdown/ViTPose-small-rfdetr` contains the model predictions from a previous run.
5. **Initialize results.tsv**: Create `results.tsv` in `projects/gp_kalman_filter` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs the post processing and the evaluation with the GP-Kalman filter on one set of model predictions. The pipeline runs with a **fixed timeout of 15 minutes**. Any evaluation that takes longer will be killed. You launch an evaluation simply as: `./projects/gp_kalman_filter/run_evaluation.sh`.

**What you CAN do:**
- Modify `projects/gp_kalman_filter/filter.py` — this is the only file you edit. Everything is fair game: the filter architecture and its steps, parameters, GP kernel, model score interpretation, etc.

**What you CANNOT do:**
- Modify the filter to non-causal. You are allowed to modify the filter logic in any way, but **it can only use past timesteps for predictions**.
- Modify `mmpose/postprocessing/filters/gp_kalman_smoother.py`. It is read-only. It contains the wrapper that runs the filter in the post processing and evaluation pipeline. It is simple for a reason: you are working on a prototype and should only modify the filter itself.
- Modify `projects/gp_kalman_filter/run_evaluation.sh`. It is read-only. It contains the evaluation harness that runs the filter on the model predictions.
- Modify `tools/postprocess_predictions.py`. It is read-only. It contains the post processing pipeline that runs the filter on the model predictions.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Edit any other file in the repository.

**The goal is simple: get the highest score.** The score is computed as `AR - 10*bMPJVE - 10*bMPJAE`, where AR is the Average Recall computed in the CocoMetric standard metric computation, bMPJVE is the EMDB dataset box size normalized joint velocity error, and bMPJAE is the EMDB dataset box size normalized joint acceleration error. You want to maximize AR and minimize bMPJVE and bMPJAE (the scalars are arbitrary to balance the scale of the three metrics). The score is used to rank the experiments and select the best ones, but you will see the individual metrics in the output to help you understand the impact of your changes.

**15 minutes** is a hard constraint. In the end this will be a realtime filter. You get the runtime of the filter as a metric to help you understand the impact of your changes. All else being equal, you want to keep the runtime as low as possible.

**Causal criterion**: The filter can only use past timesteps for predictions. This is a hard constraint.

**Changing the algorithm instead of hyperparameter tuning**: You are encouraged to make changes to the GP Kalman filter algorithm itself, rather than just tuning hyperparameters. This is because the filter is a prototype and you are trying to understand the impact of changes that can't be covered by e.g. a simple grid search. You can still tune hyperparameters, but a change to the algorithm itself is more valuable than a slightly better hyperparameter configuration.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 score improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 score improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the script as is.

## Output format

Once the script finishes it prints a summary like this:

```
Runtime: 290 s
coco/AR: 0.9127
emdb/bMPJVE: 0.0077
emdb/bMPJAE: 0.0062
score: 0.9266
```

Note that the script is configured to kill the pipeline after 15 minutes. You can read its results from the log file:

```
cat projects/gp_kalman_filter/run_evaluation.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 8 columns:

```
commit	runtime   coco/AR  emdb/bMPJVE emdb/bMPJAE   score	status	description
```

1. git commit hash (short, 7 chars)
2. Runtime in seconds
3. Average Recall (AR) computed in the CocoMetric standard metric computation
4. EMDB dataset box size normalized joint velocity error (bMPJVE)
5. EMDB dataset box size normalized joint acceleration error (bMPJAE)
6. Score (sum of AR, bMPJVE, and bMPJAE)
7. status: `keep`, `discard`, `timeout`, or `crash`
8. short text description of what this experiment tried

Example:

```
commit	runtime   coco/AR  emdb/bMPJVE emdb/bMPJAE   score	status	description
f3a5b7c	286 0.9281 0.0065 0.0054 0.8091	keep	baseline
c4d6e8f	292 0.9310 0.0062 0.0050 0.8190	keep	increase window size to 40
e6f8g1h	290 0.9123 0.0070 0.0056 0.7863	discard	use lower variance for measurement
g1h2j3k	600 0.9472 0.0042 0.0043 0.8622	timeout	increase window size to 100
k2l4m6n	278 0.9395 0.0063 0.0049 0.8275	keep	use innovation gating
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `projects/gp_kalman_filter/filter.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `./projects/gp_kalman_filter/run_evaluation.sh`
5. Read out the results: `cat projects/gp_kalman_filter/run_evaluation.log`
6. You see the metrics if the run finished successfully, or the timeout, or the error log if it crashed.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If score improved (higher), you "advance" the branch, keeping the git commit
9. If score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers, look up similar problems in the literature, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

**Hints**: The filter concept is very similar to the Kalman filter, but designed to handle the smooth but hard to model motion of human keypoints. You can use standard improvements to the Kalman filter as a reference point for your experiments. But also feel free to get wild and crazy with the filter. The keypoint score predicted by the model is a noisy proxy for measurement variance, you might need to find better ways to interpret it or use it in a different way.

As an example use case, a user might leave you running while they sleep. If each experiment takes you up to 10-15 minutes then you can run approx 4-6/hour, for a total of about 50 over one night. The user then wakes up to experimental results, all completed by you while they slept!
