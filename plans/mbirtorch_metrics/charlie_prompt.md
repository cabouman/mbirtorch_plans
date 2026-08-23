# Prompt for Charlie's Claude

Copy everything below the line into a Claude session on Charlie's machine.  It is self-contained.

---

We have a new metrics repository, `github.com/cabouman/mbirtorch_metrics`.  It is the performance
and correctness dashboard for mbirtorch, and it is a port of `gbuzzard/mbirjax_metrics` with
mbirtorch in place of mbirjax.  The code, the tooling, and 41 measured nightly runs are already
pushed to it.  Greg built and verified it on 2026-08-23.

One thing is left that only Charlie can do, because it needs admin permission on the repository.
Greg has `write`, not `admin`, so he cannot do it himself.

## The task: publish the dashboard on GitHub Pages

The repository already contains the workflow that builds and deploys the dashboard, at
`.github/workflows/pages.yml`.  That workflow has run twice and failed both times, at the same step.
The build step succeeded and produced the dashboard.  Only the `actions/configure-pages` step
failed, because GitHub Pages is not switched on for the repository.

Switch Pages on, with the source set to GitHub Actions.  Either route works.

In the web interface: open the repository's Settings, then Pages, then under "Build and deployment"
set Source to "GitHub Actions".

From the command line, if `gh` is authenticated as `cabouman`:

```bash
gh api -X POST repos/cabouman/mbirtorch_metrics/pages -f build_type=workflow
```

Then trigger the workflow again, because the two existing runs already failed:

```bash
gh workflow run pages.yml --repo cabouman/mbirtorch_metrics
```

Then confirm it worked.  Watch the run, and check that every step succeeded rather than only that
the run finished:

```bash
gh run list --repo cabouman/mbirtorch_metrics --limit 3
```

The dashboard should then be live at <https://cabouman.github.io/mbirtorch_metrics/>.  Open it and
check three things: the title reads "mbirtorch metrics", the footer reports 41 or more runs across
platforms `cpu` and `gpu`, and the History section draws three charts.

Report back whether Pages is on, whether the deploy succeeded, and whether the page loads.

## Please do not handle any access token

There is a second, separate blocker that is NOT your task.  The nightly job on Purdue's Gautschi
cluster cannot push its results to this repository yet, because the credential stored there is
scoped to the old repository.  Solving that needs a new GitHub access token.

Do not create, request, read, paste, or store any token.  That is a step for a person to do by hand.
If Charlie wants to solve it from his side, the manual step is: create a fine-grained personal
access token whose resource owner is `cabouman`, scoped to the `mbirtorch_metrics` repository only,
with Repository permissions set to Contents: Read and write, and give it to Greg through whatever
channel they normally use for secrets.  Greg's alternative, which needs nothing from Charlie, is a
classic token with the `repo` scope created under his own account.

## One thing worth deciding later, not now

The two metrics repositories have different owners.  `mbirjax_metrics` belongs to `gbuzzard` and
`mbirtorch_metrics` belongs to `cabouman`.  That asymmetry is what caused both blockers: Greg cannot
change repository settings, and a fine-grained token he creates cannot reach a repository owned by
another user account.

Moving `mbirtorch_metrics` under a shared organization, or under the same owner as the other
repository, would remove both problems permanently.  Raise it with Charlie and Greg as a question.
Do not move or transfer anything yourself.
