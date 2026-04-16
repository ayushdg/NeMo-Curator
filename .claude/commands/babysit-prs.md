Babysit all PRs listed in `.claude/pr-watchlist.md`. For each PR, check its status and take the appropriate action.

The watchlist and the flaky-reruns tracker are per-user (untracked). Each teammate maintains their own list of PRs they want babysat.

## Read the watchlist

Read `.claude/pr-watchlist.md`. Extract PR numbers from lines matching `- <number>`. If the watchlist is empty, report "No PRs to babysit" and stop. If the file doesn't exist, copy it from `.claude/pr-watchlist.md.example` and report "No PRs to babysit — created an empty watchlist at .claude/pr-watchlist.md for you to populate". Do the same for `.claude/pr-flaky-reruns.md` from `.claude/pr-flaky-reruns.md.example` the first time a flaky re-run needs to be recorded.

## Identify the current user

Run `gh api user --jq .login` once and cache the result for this run. This login is the "owner" identity used in Step 2 — PRs authored by this user are treated as the user's own.

## For each PR, execute these steps in order

### Step 1: Fetch PR info

```bash
gh pr view <N> -R NVIDIA/NeMo-Curator --json number,title,state,author,headRefOid,mergeable,mergeStateStatus,updatedAt
```

### Step 2: Classify the PR author

- `author.login == "dependabot[bot]"` → **dependabot**
- `author.login == <current user from the lookup above>` → **owner** (user's own PR)
- anything else → **external**

### Step 3: Handle merged/closed PRs

- If `state == "MERGED"`: remove from watchlist, report as merged.
- If `state == "CLOSED"`: flag to user ("PR was closed without merging"), remove from watchlist.
- If `state == "OPEN"`: continue to Step 4.

### Step 4: Check CI status

Get the latest CICD workflow run for this PR's branch:

```bash
gh run list -R NVIDIA/NeMo-Curator --branch "pull-request/<N>" --workflow "CICD NeMo Curator" --limit 5 --json databaseId,status,conclusion,headSha
```

Use the results to determine the situation:

#### Case A: No CICD run exists for the current head SHA AND the PR is dependabot or external

CI has not been authorized for the latest commit. Before commenting, check that we haven't already commented `/ok to test` for this SHA:

```bash
gh api repos/NVIDIA/NeMo-Curator/issues/<N>/comments --jq '.[-10:][].body'
```

If no `/ok to test <current_SHA>` comment exists, authorize CI:

```bash
gh pr comment <N> -R NVIDIA/NeMo-Curator --body "/ok to test <headRefOid>"
```

#### Case B: Latest CICD run failed

First, count how many completed CICD runs exist for the **same head SHA** with `conclusion == "failure"`. This tells us how many times CI has already been attempted for this commit.

Then inspect the failure logs:

```bash
gh run view <run_id> -R NVIDIA/NeMo-Curator --json jobs --jq '.jobs[] | select(.conclusion == "failure") | {name, conclusion}'
gh run view <run_id> -R NVIDIA/NeMo-Curator --log-failed 2>&1 | tail -200
```

Analyze the logs to classify the failure:

**PR-related failure indicators** (flag immediately, do NOT re-run):
- Import errors or syntax errors in modules the PR modified
- Test assertion failures in test files that correspond to code changed by the PR
- Linting or formatting failures on files in the PR diff
- Type errors in PR-changed code

If it looks PR-related, report: "CI failure appears related to PR changes: <brief summary of error>"

**Unrelated/flaky failure indicators** (auto re-run):
- Timeouts (job exceeded time limit)
- Network/connectivity errors (pip install failures, download timeouts)
- Infrastructure errors (runner issues, Docker pull failures)
- Failures in test folders completely unrelated to the PR's changes
- Known flaky patterns (race conditions, intermittent assertion errors in unrelated tests)
- `EDQUOT`, `ENOMEM`, or other system resource errors

Re-run decision based on attempt count:
- **Fewer than 3 prior failures for this SHA**: Re-run failed jobs:
  ```bash
  gh run rerun <run_id> -R NVIDIA/NeMo-Curator --failed
  ```
- **3+ prior failures for this SHA**: Do NOT re-run. Flag to user: "Flaky failure persisting after 3 re-runs — needs manual inspection"

**After any flaky re-run or flaky flagging**, update the tracker in `.claude/pr-flaky-reruns.md`:
- One row per PR. If the PR already has a row, increment "Flaky Re-runs", update "Failed Jobs" and "Last Re-run Date".
- If not, add a new row.
- When a PR is removed from the watchlist (merged/closed), keep its row in the tracker for historical metrics.

#### Case C: CICD run is in progress

Report as "CI running", take no action.

#### Case D: CICD run passed

Report as "CI green", take no action.

### Step 5: Check branch freshness

Look at `mergeStateStatus` from Step 1. If the branch is behind (`BEHIND` or `DIRTY`):

**Owner PR (current user)**:
Auto-update the branch (equivalent to the "Update branch" button in GitHub UI):
```bash
gh api repos/NVIDIA/NeMo-Curator/pulls/<N>/update-branch -X PUT
```

**Dependabot PR**:
First check recent comments to avoid spamming:
```bash
gh api repos/NVIDIA/NeMo-Curator/issues/<N>/comments --jq '.[-5:][].body'
```
- If no recent `@dependabot rebase` comment: comment `@dependabot rebase`
  ```bash
  gh pr comment <N> -R NVIDIA/NeMo-Curator --body "@dependabot rebase"
  ```
- If already commented `@dependabot rebase` and branch is still behind: flag to user — "Dependabot hasn't rebased after being asked, may need manual intervention"

**External PR**:
Flag to user only: "PR is behind main — author needs to update their branch". Never auto-update.

### Step 6: Report summary

After processing all PRs, print a concise status table. Example format:

```
PR Babysitter Report
────────────────────
#1234 (CVE fixes)           ✓ CI green, up to date
#1235 (bump foo 1→2)        ↻ Re-ran failed CI (attempt 1, flaky timeout in stages-video)
#1236 (new feature)         ⚠ CI failure looks PR-related: ImportError in new_module.py
#1237 (bump bar)            ✗ Removed — merged
#1238 (bump baz)            ⚠ Behind main — flagged (external PR)
#1239 (dep update)          → Commented /ok to test abc1234
```

### Step 7: Update the watchlist

Edit `.claude/pr-watchlist.md` to remove any PRs that were merged or closed.
