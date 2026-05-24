# vendor/ — third-party libraries imported as git subtrees

We vendor an upstream library here (instead of pinning a PyPI version) when:

- the published release is broken in a way that blocks us, and
- the upstream is unmaintained or slow enough that waiting for a fix isn't workable.

Each entry is a `git subtree --squash` import, installed as an editable local
dep through `pyproject.toml`'s `[tool.uv.sources]`. Local patches go directly
in the vendored tree; tag them with a comment beginning `LOCAL PATCH
(agent-driven-investment vendor):` so the next sync knows what to re-apply.

## Current vendored libraries

### jugaad-data

- Source: `https://github.com/jugaad-py/jugaad-data` (master)
- Imported via: `git subtree add --prefix=vendor/jugaad-data https://github.com/jugaad-py/jugaad-data.git master --squash`
- Upstream version at import: `0.33.1`
- Installed as: editable, via `[tool.uv.sources]` jugaad-data = `{ path = "vendor/jugaad-data", editable = true }`
- Why vendored: the index history endpoint
  (`niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString`) changed
  contract around mid-2026 and now requires a `cinfo` JSON-string parameter
  the unmodified library does not send. Result was HTTP 500 from the server
  surfacing as `KeyError: 'd'` in `_index`. Discovered + diagnosed during
  data_pipelines v1.7. See
  `docs/data_pipelines/V1_IMPLEMENTATION_PLAN.md` §"Implementation findings —
  Known upstream limitations".

#### Local patches (apply to vendor/jugaad-data/jugaad_data/nse/history.py)

Search the file for `LOCAL PATCH` to find them. As of 2026-05-24:

- `NSEIndexHistory._index` — wraps params in `{'cinfo': json.dumps({...})}`
- `NSEIndexHistory._index_pe` — same change for the sibling PE/PB endpoint

#### Syncing from upstream

```bash
git subtree pull --prefix=vendor/jugaad-data \
    https://github.com/jugaad-py/jugaad-data.git master --squash
```

After a pull, re-grep `vendor/jugaad-data` for `LOCAL PATCH` markers; if the
pull dropped any, re-apply from this file's history or git log on the merge
commit.
