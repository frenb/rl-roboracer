# Fixing the Experiments tab

The tab could only express 24 of the trainer's 39 experiment-design fields,
and it dropped the other 15 silently.

**Status — 2026-09-20: Part 1 is complete (Steps 1, 2, 2a, 3).** The mirror
is re-synced, the live endpoint serves all 6 sections and 39 fields, and a
round-trip through `/add_experiment_design` confirms the write path now
keeps them — the silent drop is closed. Parts 2-5 remain. Note that
`curriculum_stages` is now served but still renders as a number box, so
Part 4 is no longer hypothetical — see the end of Part 1.

Two further copies of the schema turned out to be stale for the same reason
as the JS mirror, both found while verifying Steps 1-2: the read-only
override tables in `jobs.html` / `models.html` (Step 2a, fixed) and the
canonical `experiment-default` document in Mongo (Step 10, fix written but
waiting on a trainer restart).

**Parts 2, 3 and 4's Step 7 are also done.** Every field is reachable and
authorable, an unrecognised key is a 400 rather than a silent drop, and the
dashboard now serves the trainer's own schema from Mongo rather than a
hand-copied mirror — so the original drift cannot recur. Step 8 is done too.
**Remaining: Step 9** (audit the collection), and verifying **Step 10**
after the next trainer restart.

Read Part 0 first. The single most useful fact is that the Python module
already exports a function returning exactly the shape the dashboard serves,
so most of this is deleting a duplicate rather than writing new code.

Sibling docs:

- Field semantics and the trainer side: `rl_agent/experiment_designs.py`.
- How a design reaches a run: "Selecting the course per job" in
  [`../README.md`](../README.md).

---

## Part 0 — What is actually broken

Measured 2026-09-20, *before* Part 1. This part is kept in the present tense
as the record of what was wrong and why; Steps 1-2 have since fixed the
mirror and the endpoint.

### A hand-maintained mirror that drifted

`dashboard/src/server.ts` carries a JS copy of the trainer's schema with this
comment above it:

> IMPORTANT: the EXPERIMENT_DESIGN_SCHEMA constant below MUST be kept in sync
> with `rl_agent/experiment_designs.py::SCHEMA`. [...] When you add a field
> there, add it here too.

That instruction was not followed. Python defines **39 fields in 6 sections**;
the mirror has **24 fields**. The 24 are a strict subset, so nothing is
misnamed — 15 fields are simply absent:

| Field | Type | Trainer default |
|---|---|---|
| `env_discount` | float | 0.9 |
| `eval_time_fraction` | float | 0.25 |
| `eval_train_interval_sec` | int | 0 |
| `awac_lambda` | float | **0.0** |
| `awac_beta` | float | 1.0 |
| `awac_weight_clip` | float | 20.0 |
| `awac_lambda_decay_steps` | int | 0 |
| `curriculum_stages` | json | None |
| `curriculum_start_stage` | int | 0 |
| `corner_radius` | float | 10.0 |
| `curvature_difficulty` | float | 0.0 |
| `chicanes_north` / `_east` / `_south` / `_west` | int | 0 |

Three whole feature families are unreachable from the UI: AWAC, the
curriculum, and track geometry.

### Why the silent drop is dangerous rather than merely annoying

`/add_experiment_design` iterates the mirror and copies only fields it knows,
so anything else in the request body vanishes without a warning — the request
still returns success. Combined with the defaults above, a design authored in
the UI and *named* for AWAC would train with `awac_lambda = 0.0`, which is
AWAC switched off. Two more quiet substitutions: `env_discount` becomes 0.9
where every AWAC design in this project uses 1.0, and `eval_time_fraction`
becomes 0.25 where they use 0.1.

This is not hypothetical. It is how this document came to exist: an attempt to
create a curriculum-free clone of an existing design lost 8 of 18 fields, and
the loss was only caught by diffing the stored document against its source.

### What is *not* broken

`/update_experiment_design` builds a `$set` containing only known fields, so
editing a design through the UI leaves unknown fields untouched. It cannot
repair them, but it never deletes them. A check of the collection agrees:
**10 designs carry at least one of the 15 fields** (all written directly to
Mongo by agents, including every AWAC and curriculum design), and **14 carry
none**. No stored design has been damaged.

### Three tabs consume the endpoint

`/get_experiment_design_schema` is fetched by `experiment_designs.html`
(which renders its entire form from it), `jobs.html:1225` and
`models.html:1237`. Fixing the constant fixes all three at once.

### The form renderer cannot express a JSON field

`_renderFieldRow` in `experiment_designs.html` has exactly two branches:
`bool` becomes a checkbox, and **everything else becomes
`<input type="number">`**. There is no text or textarea path. So
`curriculum_stages`, whose type is `json`, would render as a number box that
cannot hold `[{...},{...}]`.

This is why Part 1 alone is not sufficient: re-syncing the schema yields 14
newly usable fields and one that renders as a broken control. **As of Step 2
this is the live state** — the backend now accepts `curriculum_stages`, but
the form still offers a number spinner for it, so Part 4 is the difference
between "accepted by the API" and "authorable in the UI".

### The fix is mostly deletion

`rl_agent/experiment_designs.py` already exports `get_schema_for_endpoint()`,
whose docstring reads "JSON-serialisable view of SCHEMA for
`/get_experiment_design_schema`". It returns a list of
`{kind: 'section', label}` and `{kind: 'field', name, type, default, min,
max, doc, paper_ref, kwarg}` entries — 6 sections and 39 fields, byte-for-byte
the shape the dashboard serves today.

The Python side was built to feed this endpoint. The JS mirror is a stopgap
that outlived its excuse. The excuse, per the same comment, is that the
dashboard container has no Python and shouldn't depend on `docker exec` —
which Part 2 respects.

Note there is no CI in this repo (`.github/workflows` does not exist), so a
"test that the two schemas match" would never run. That is why Part 2
removes the duplication instead of guarding it.

---

## Part 1 — Stop the bleeding

Makes the tab usable today. Roughly an hour.

### Step 1 — Regenerate the constant from Python

**Do this.** Generate the TypeScript block from the authoritative schema and
splice it over the `EXPERIMENT_DESIGN_SCHEMA` literal in
`dashboard/src/server.ts`. A plain `json.dumps` is *not* pasteable — it emits
double-quoted keys and no trailing commas, so it would have to be hand-edited
into the file's style, which is the retyping this step exists to avoid. Emit
the source lines directly instead:

```powershell
$py = @'
import sys
sys.path.insert(0, "/python_ws/src")
from experiment_designs import get_schema_for_endpoint

def q(s):
    if s is None:
        return "null"
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"

def num(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v)

entries = get_schema_for_endpoint()
fields = [e for e in entries if e["kind"] == "field"]
wn  = max(len(q(f["name"]))      for f in fields) + 2
wt  = max(len(q(f["type"]))      for f in fields) + 2
wd  = max(len(num(f["default"])) for f in fields) + 2
wmi = max(len(num(f["min"]))     for f in fields) + 2
wma = max(len(num(f["max"]))     for f in fields) + 2

out = ["  const EXPERIMENT_DESIGN_SCHEMA: any[] = ["]
for e in entries:
    if e["kind"] == "section":
        out.append("    { kind: 'section', label: %s }," % q(e["label"]))
        continue
    out.append(
        "    { kind: 'field', name: %s type: %s default: %s min: %s max: %s doc: %s, paper_ref: %s, kwarg: %s },"
        % ((q(e["name"])      + ",").ljust(wn),
           (q(e["type"])      + ",").ljust(wt),
           (num(e["default"]) + ",").ljust(wd),
           (num(e["min"])     + ",").ljust(wmi),
           (num(e["max"])     + ",").ljust(wma),
           q(e["doc"]), q(e["paper_ref"]), q(e["kwarg"])))
out.append("  ];")
sys.stdout.write("\n".join(out))
'@
$py | docker compose exec -T sim-controller python - > schema_block.ts.txt
```

`q()` escapes embedded apostrophes, which matters because several `doc`
strings contain them. Splice by line number rather than by string match —
the literal is ~45 very long lines and an exact-match edit is fragile. Delete
the generated file afterwards; it is a build artifact, not a source file.

**Why it matters.** Hand-editing 15 entries is how the drift happened. Paste
generated output; do not retype it.

**You are done when.** The constant holds 6 sections and 39 fields.

**Status — done 2026-09-20.** 48 insertions, 25 deletions in `server.ts`. The
comment above the constant was also rewritten: it used to say "when you add a
field there, add it here too", which is the instruction that failed, and now
says to regenerate rather than hand-edit.

### Step 2 — Rebuild and verify the endpoint

**Do this.** `server.ts` compiles via `tsc` to `dashboard/build/server.js`
(`npm run build`, which `prestart` invokes), so restart the dashboard and
check what it serves:

```powershell
cd dashboard; npx tsc -p tsconfig.json --noEmit; cd ..   # ~20s, catches syntax errors
docker compose restart dashboard                          # ~45s incl. rebuild
(Invoke-RestMethod http://localhost/get_experiment_design_schema).fields.Count
```

TypeScript is already in `dashboard/node_modules`, so the `--noEmit` check
runs on the host and is worth doing before paying for a container restart.
The restart needs no separate build step: the service is
`sh -c "cd /dashboard && npm start"`, and `prestart` compiles.

Note the response is `{"fields": [...]}`, not a bare array — hence the
`.fields` above. Filtering the top-level object instead silently matches
nothing and looks like an empty schema.

**Why it matters.** Editing `server.ts` without a rebuild changes nothing;
the container runs the compiled output.

**You are done when.** The count is 45 (39 fields + 6 sections), and the
Experiments tab shows the new AWAC, curriculum and track-geometry sections.

**Status — done 2026-09-20.** Endpoint serves 45 entries (6 sections, 39
fields); all 15 previously-missing fields verified present with their
trainer defaults and bounds. `tsc --noEmit` clean.

Trap 5 checked, and it found one thing (fixed, see below): `jobs.html` and
`models.html` don't build inputs from the schema — both render the same
read-only "field | override | default" table — so the `json`-renders-as-a-
number-spinner problem doesn't reach them. Verified against the live
endpoint and the real `AWAC + No-BC + curriculum (5-stage)` design: 14 of
the 15 new fields display correctly, and the two pages agree cell-for-cell.

### Step 2a — Read-only modals can't print an object either

Not in the original plan; found while checking trap 5.

Both modals rendered the override cell with `String(val)`, which is fine for
the 38 scalar fields and produces
`[object Object],[object Object],…` for `curriculum_stages`. This was a small
regression from Step 2: before the re-sync the field wasn't in the schema, so
no row was drawn at all.

Note this is *not* covered by Part 4 — Step 7 fixes `_renderFieldRow` in
`experiment_designs.html`, a different renderer in a different file. Three
pages display these fields and each has its own code path.

**Done 2026-09-20.** Added `_formatOverrideValue` / `formatOverrideValueLocal`
next to the existing escape helper in each file: scalars are unchanged,
objects and arrays become a `<details>` disclosure over a `<pre>` of
`JSON.stringify(val, null, 2)`. A summary count keeps the row one line tall,
which matters because the 5-stage array is 744 characters. Verified the
disclosed JSON parses back to the identical 5-stage array.

### Step 3 — Round-trip a design

**Do this.** Create a design in the UI with `awac_lambda = 0.5` and
`env_discount = 1.0`, then read the stored document back and confirm both
fields are present with those values.

**Why it matters.** The endpoint reporting 39 fields does not prove the write
path keeps them; the copy loop is what dropped them.

**You are done when.** Both fields survive the round trip.

**Status — done 2026-09-20, 8 of 8 fields survived.** Posted via the API,
which exercises the same copy loop the UI does. Design
`6ab0d1a64bc11521c6641124`, "Round-trip check (plan Step 3)", left in the
collection deliberately; delete it whenever.

`env_discount` 1.0, `awac_lambda` 0.5, `awac_beta` 2.0, `eval_time_fraction`
0.1, `curriculum_start_stage` 2, `corner_radius` 14.0 and `chicanes_north` 1
all came back with the right values and BSON type `number` — every one of
them a field that would have vanished before Step 1. `curriculum_stages`
came back as a real 2-element array of objects, so the *API* stores JSON
correctly; only the form control can't author it (Part 4).

**This also confirms Part 3 is still needed.** The same post included a
deliberate typo, `awac_lamda: 0.9`. It is absent from the stored document
and the endpoint returned `{"acknowledged": true}` with the new `_id`. A
caller still cannot distinguish "saved" from "silently discarded" — exactly
the failure that motivated this plan, now reproduced on demand.

Part 1 is complete.

---

## Part 2 — Remove the duplication

So it cannot drift again. The constraint is that the dashboard container has
no Python interpreter.

### Step 4 — Have the trainer publish the schema

**Do this.** At trainer startup, alongside the existing
`_seed_canonical_reward_designs()` and `_seed_canonical_experiment_design()`
calls, upsert the schema into Mongo:

```python
db.schema_registry.update_one(
    {"_id": "experiment_design"},
    {"$set": {"fields": get_schema_for_endpoint(),
              "updated_at": datetime.datetime.now(datetime.timezone.utc)}},
    upsert=True)
```

**Why it matters.** The trainer already seeds canonical documents into Mongo
on every start, so this adds no new mechanism and no new dependency. Mongo is
the channel these two containers already share. The schema becomes current
the moment the trainer runs, with no build step and no human in the loop.

**You are done when.** `db.schema_registry.findOne({_id:'experiment_design'})`
returns 45 entries.

**Status — done 2026-09-20, effective at the next trainer restart.**
`_publish_experiment_design_schema()` in `robotaxi.py`, called from
`run_jobs_loop` right after the two seed calls. Deliberately unversioned and
unconditional, unlike `_seed_canonical_experiment_design` beside it: this is
a derived cache of SCHEMA, not user data, and a version guard on it is
exactly what left the canonical Default stale for three weeks (Step 10).
Failures are caught and logged, because the dashboard's fallback makes this
non-fatal.

The document exists **now** — I ran the identical upsert by hand so Part 2
could be verified today without restarting the trainer mid-job. The restart
only changes it from "published once" to "republished every boot". Note the
function itself has therefore not executed yet; `robotaxi.py` was not
imported to run it, because that pulls in TensorFlow and would contend for
the GPU the live job is using.

### Step 5 — Serve from Mongo, keep the constant as fallback

**Do this.** Change the endpoint to prefer the stored copy:

```typescript
app.get('/get_experiment_design_schema', (req, res) => {
  dbo.collection('schema_registry').findOne({_id: 'experiment_design'},
    (err, doc) => {
      if (!err && doc && Array.isArray(doc.fields) && doc.fields.length) {
        res.json({ fields: doc.fields, source: 'trainer' });
        return;
      }
      res.json({ fields: EXPERIMENT_DESIGN_SCHEMA, source: 'fallback' });
    });
});
```

Then reduce the comment above the constant to say it is a stale-tolerated
fallback for a database with no trainer-published schema, not a thing to keep
in sync by hand.

**Why it matters.** The `source` field makes the failure visible: if the tab
is ever driven by the fallback, that shows in the response instead of
manifesting months later as a dropped field. The write paths in Steps 6 keep
using the served list, so they inherit the fix automatically.

**You are done when.** The endpoint reports `source: 'trainer'`, and deleting
the `schema_registry` document makes it report `source: 'fallback'` without
erroring.

**Status — done 2026-09-20, 14 of 14 checks pass.**

**This step was bigger than written, and the sentence above about the write
paths inheriting the fix was wrong.** By the time Part 2 started the
constant had *four* consumers, not one: the GET endpoint, Step 6's
`unknownExperimentDesignKeys`, and the copy loops in both write handlers.
Changing only the endpoint would have been actively harmful — in exactly
the scenario this part exists for, a trainer publishing a new field, the
form would render it, the user would save it, and the validator would
reject it with a 400 naming a key the same server had just advertised.

So the shape is a single resolver, `withExperimentDesignSchema(cb)`, which
prefers `schema_registry` and falls back to the constant. All four go
through it; `EXPERIMENT_DESIGN_SCHEMA` is now referenced in exactly one
place, the fallback line inside that resolver. Both write handlers were
restructured to run their bodies inside the callback. This suits the
codebase: the mongodb driver is 4.2.2 (callbacks still supported, removed
in 5.x) and `server.ts` has zero `async` handlers, so no new idiom.

The comment above the constant now says FALLBACK ONLY and that going stale
is acceptable by design.

Verified, including the case that motivated the resolver: with a
`future_knob` field published only to `schema_registry` and absent from the
constant, the endpoint serves it, the write path accepts it, and the value
is stored — then with the registry document deleted, the same post is
rejected 400 against the fallback list. The write path really does follow
whichever list is live. Fallback also serves 45 entries and accepts normal
writes, and `source` flips back to `trainer` on restore.

**Testing note.** The first run of the verification script crashed partway
(it compared a string `insertedId` against an `ObjectId`), which both left
its test field in `schema_registry` and made its cleanup a silent no-op.
The next run then captured the contaminated document as its baseline and
reported three failures that were entirely self-inflicted. If you write a
checker for this, restore state in a `finally` and delete by a distinctive
name, not by an id you round-tripped through JSON.

---

## Part 3 — Make a silent drop impossible

Worth doing even with the schemas in sync, because it also catches typos.

### Step 6 — Report ignored keys instead of swallowing them

**Do this.** In both `/add_experiment_design` and
`/update_experiment_design`, collect any body key that is neither schema
metadata (`name`, `description`, `author`, `archived`, `fields`) nor a known
field, and return HTTP 400 listing them.

**Why it matters.** The bug that motivated this plan was not that the schema
was stale — it was that the API accepted 18 fields, stored 10, and answered
"success". A caller cannot distinguish "saved" from "silently discarded". A
400 makes the next occurrence a one-line error instead of an experiment whose
results quietly mean something else.

**You are done when.** Posting a design with `awac_lamda` (typo) is rejected
naming that key, and a correct post still succeeds.

**Status — done 2026-09-20, 8 of 8 checks pass.**

`unknownExperimentDesignKeys()` in `server.ts` diffs the body against the
schema's field names plus a metadata allow-list, and `rejectUnknownKeys()`
answers with 400 `{error, unknown_keys, hint}`. Both write endpoints call it
right after their existing required-field check.

Three decisions worth knowing:

- **`version`, `created_at` and `updated_at` are allowed and ignored**, on
  top of the metadata the step lists. They're server-managed, and rejecting
  them would break any read-modify-write caller that echoes a stored
  document back. There is a passing test for exactly that shape.
- **The nested `{fields: {...}}` shape is checked too**, and its bad keys
  are reported namespaced (`fields.not_a_field`) so the caller can tell
  which half of the body was wrong.
- **The UI had to change as well.** `saveDesign` threw
  `new Error('HTTP ' + res.status)`, so a 400 would have surfaced as a bare
  "HTTP 400" toast and hidden the key name — the whole point of the step. A
  new `_errorFromResponse()` lifts `error` and `hint` off the JSON body.

Verified: the `awac_lamda` typo from Step 3 is now rejected with
`unknown experiment-design field(s): awac_lamda`; several bad keys are all
named at once; `/update_experiment_design` rejects too; and flat, nested and
read-modify-write posts of legitimate bodies still return 200. Test designs
cleaned up, the Step 3 and Step 7 ones kept.

**Scope limit.** This guards the HTTP path only. `madscientist`'s
orchestrator writes designs with `db.experiment_designs.insert_one()`
(`orchestrator.py:245`), spreading `**overlay` straight into the document,
so it can still store any key it likes without validation. That is also why
this change was safe to ship mid-run: the dashboard UI is the only HTTP
caller of these two endpoints.

---

## Part 4 — Render the JSON field properly

### Step 7 — Give `curriculum_stages` a real control

**Do this.** Add a third branch to `_renderFieldRow`: for `type === 'json'`,
emit a `<textarea>` rather than `<input type="number">`, parse it with
`JSON.parse` on change, and show an inline error when it doesn't parse.
`_onFieldInput` needs a matching branch so the value is stored as parsed JSON
rather than coerced to a number.

**Why it matters.** Without this, Part 1 hands the user a numeric spinner for
a five-element array of stage objects. Better to render nothing than a
control that silently produces `NaN`.

**Confirmed live 2026-09-20** (screenshot of the Track / Environment
section). Two details are worse than this step assumed, both because the
control is `type="number"` rather than a plain text input:

- It does not produce `NaN`. An `input[type=number]` sanitises any
  non-numeric value to the empty string, and `_onFieldInput` treats empty as
  "use trainer default" and **deletes the key from `editorState.overrides`**.
  So pasting a stage array and saving is a silent no-op, not a visible error.
- `_populateFormFromState` assigns `String(val)` on load, so opening a design
  that *already has* stages shows an empty box — indistinguishable from
  "not set". The value survives in `editorState.overrides` and is saved back
  intact, but the form is lying about what is stored.

The placeholder also reads `default: ` with nothing after it, because this is
the only field whose default is `None`.

**You are done when.** You can paste the 5-stage array from the
"AWAC + No-BC + curriculum (5-stage) + reward_scale1" design into the form,
save, and read back an array of 5 objects — not a string and not null.

**Status — done 2026-09-20. Done-when verified: PASS.** Took five edits in
`experiment_designs.html` plus a CSS block, not the two this step predicted.

- `_renderFieldRow` gained a `json` branch: a `<textarea>` plus an error
  slot, and the row gets `ed-field-row--json` so the label top-aligns.
- `_onFieldInput` validates with `JSON.parse` on every keystroke and shows
  the message inline; whitespace-only counts as "no override".
- `_populateFormFromState` re-validates after filling, so reopening a
  design with bad stored JSON flags it immediately.
- The save path parses the text into real JSON. On a parse failure it
  **aborts the save** and toasts, rather than `continue`-ing past the field
  — dropping it silently is the exact behaviour this plan exists to kill.
- CSS for the textarea, the invalid state and the error text.

**A sixth edit fixed a data-corruption bug this step did not know about.**
`selectDesign` populated the override map with `String(v)`, which turns a
stage array into `"[object Object],[object Object]"`. Because the save path
writes that map straight back, editing *any other field* on a curriculum
design and pressing Save would have overwritten the stored stages with that
string. It now stores `JSON.stringify(v, null, 2)`.

This corrects the Step 7 note above, which said the value "survives in
`editorState.overrides` and is saved back intact". It did not.

Verified end to end: the 5-stage array from the reward_scale10 design,
pushed through the new load transform and the new save transform and posted
to the API, reads back as an identical 5-element array of objects. The
inline script parses clean. Design `6ab0d3b54bc11521c6641125`, "Step 7
textarea round-trip", is left in the collection — open it to see the
textarea populated with real stages.

**Not verified:** the rendering itself. The logic and the JSON round trip
are proven, but no browser has drawn this textarea yet.

### Step 8 — While you are there, warn about inert curricula

**Do this.** When a design carries `curriculum_stages`, note in the form that
stages only take effect on a `TrackGen*` gym.

**Why it matters.** The trainer already warns about this at run time; saying
it at authoring time is cheaper than discovering it in a log four hours in.

**Status — done 2026-09-20, 8 of 8 checks pass.**

Scoped wider than the step describes, because the constraint is wider:
`curriculum_stages` is not the only inert field. All eight knobs in the
Track / environment section — the two curriculum ones plus `corner_radius`,
`curvature_difficulty` and the four `chicanes_*` — are applied by the same
`TrackGenerator` and ignored by the same gyms. So the caveat is a
section-level banner rather than a per-field note on `curriculum_stages`.

It renders always, not only when stages are set. A design with
`corner_radius` alone has the identical problem, and the warning is most
useful *before* you fill anything in.

Which section gets it is derived from the schema (the one containing
`curriculum_stages`) rather than matched on the label text, so renaming the
section won't silently drop the note. Styling reuses the existing
read-only-canonical banner's classes; no new CSS.

The rule it states is the trainer's actual one, `_TRACKGEN_GYM_PREFIX`: a
case-insensitive `trackgen` prefix on the gym name. Of the 45 registered
gyms, 16 qualify; every `wCourseJetRacer*` build — including the one the
current fly-brain job runs on — does not.

Verified: the note is defined once and pushed once, targets
`Track / environment (curriculum)`, that section contains all 8 geometry
knobs and nothing else, and the markup is balanced.

**Not verified:** how it looks. No browser has rendered it.

---

## Part 5 — Confirm nothing was damaged

### Step 9 — Audit the collection

**Do this.** For every design, check whether it carries the 15 fields, and
sanity-check that no design whose *name* mentions AWAC or a curriculum is
missing the matching fields.

**Why it matters.** The current answer is clean — the 10 designs that use
these fields all have them, because they were written directly to Mongo
rather than through the UI. Re-run this after Part 1 so you can tell a
pre-existing gap from one this work introduced.

**You are done when.** No design's name promises a feature its fields don't
configure.

**Status: done 2026-09-21.** The audit is now a re-runnable script at
`dashboard/tools/audit_experiment_designs.js`:

```
docker compose exec -T dashboard node /dashboard/tools/audit_experiment_designs.js
```

It resolves each design the way the trainer does — schema defaults overlaid
with the design's own fields — and reports three classes of finding: a NAME
that promises something the effective config doesn't deliver, a CONFIG that
is internally inconsistent, and an INHERIT where a derived `auto:*` design
dropped fields from its stated base.

**Part 1 introduced no damage.** 12 of 26 designs carry at least one of the
15 fields, up from 10 of 24 — the two additions are the designs Steps 3 and 7
created, and every pre-existing design still carries what it did before.

**But the audit found an unrelated and more serious bug**, described in Step
11 below. Three designs' names promise settings their fields don't configure,
and the cause is not the schema drift this plan was written about.

The only other finding is `Round-trip check (plan Step 3)`, the throwaway
design Step 3 created to prove the write path works. Its values were chosen to
be distinctive, not coherent, so it correctly trips two CONFIG checks. It has
never been attached to a job. Archive or delete it once Part 1 is signed off.

### Step 11 — Derived `auto:*` designs silently ignore their base

Not in the original plan; found 2026-09-21 by the Step 9 audit.

`_derive_design()` in `rl_agent/madscientist/orchestrator.py` builds an arm's
design as metadata plus `**overlay`, where the overlay is only the arm's
explicit `experiment_design_fields`. It loads `base_doc` but reads nothing
from it except `name`, for the description string. So a derived arm does
**not** inherit its base's settings; any field the arm doesn't restate falls
back to `main()`'s stock defaults.

The docstring says the opposite — "Arms with no overlay still get a derived
design, same fields as the base" — which is presumably the intent. Either the
docstring or the code is wrong, and the code is what ran.

All 11 `auto:*` designs drop base fields, but for 8 of them the base is
`Default`, whose values are identical to the trainer defaults they fall back
to, so nothing changes. Three are material:

| design | jobs / models | effect |
| --- | --- | --- |
| `auto:219d1526:awac_base` | 0 / 0 | never ran |
| `auto:219d1526:awac_wide_critic_1024` | 0 / 0 | never ran |
| `auto:c640cc4c:base` | 5 / 13 | **ran, and confounded its proposal** |

Proposal `6a39e7f254e33053c640cc4c` (status `training`) tests: "on the current
best-performing configuration (Demo-protected + discount1.0), enabling AWAC
with `awac_lambda=0.3` and `awac_beta=0.5` (vs plain SAC, `awac_lambda=0.0`)
will increase mean eval `avg_return` by at least 15% over 5 seeds."

The treatment arm `awac_sharp` restates all 11 of its fields explicitly, so it
is correct. The control arm `base` carries only `num_iterations`, so instead of
"the same configuration minus AWAC" it ran stock defaults:

| field | control intended | control actually ran |
| --- | --- | --- |
| `replay_buffer_capacity` | 300000 | 75000 |
| `demo_min_keep` | 50000 | 0 |
| `demo_sample_ratio` | 0.25 | 0 |
| `env_discount` | 1.0 | 0.9 |
| `eval_time_fraction` | 0.1 | 0.25 |

The paired-seed comparison therefore measures AWAC *plus* demo protection plus
a 4x replay buffer plus a longer horizon, against none of them — not AWAC
alone. Whatever this proposal concludes about AWAC will not be supported by
what it ran.

**Correction, 2026-09-21.** An earlier version of this paragraph also claimed
the differing `eval_time_fraction` meant the two arms' `avg_return` values were
"measured over different eval durations". That is wrong, and the error is worth
recording because the parameter's name invites it. `eval_time_fraction` sets
the *cadence* of evals — the loop trains `(1-frac)/frac` times each eval's
wall-clock before the next one — not the length of an eval. Every eval runs
`num_eval_episodes` (10 here, identical across both arms) to termination, so
each individual measurement is comparable. What it does change is how many eval
points a run produces and how much wall-clock goes to evaluating, which biases
any *maximum over evals* statistic toward the arm that evaluated more often.
That is a real but second-order effect, unlike the four genuine confounds above.

The two `auto:219d1526` arms are a second, independent problem: both carry
only `num_iterations`, so after the missing inheritance they are byte-identical
to each other, and the arm named `awac_wide_critic_1024` never set a critic
width. That proposal is marked `done` having compared two identical arms.

**Suggested fix**, not yet applied — it needs a decision because proposal
c640cc4c is mid-flight:

1. In `_derive_design()`, seed `derived` with the base's tunable fields before
   applying `**overlay`, so the overlay genuinely overlays the base. Exclude
   metadata keys (`_id`, `name`, `description`, `version`, `archived`,
   `create_date`, `proposal_id`, `base_design_id`).
2. Re-run proposal c640cc4c's control arm. This is cheaper than it sounds:
   the arm is already incomplete for an unrelated reason, so nothing usable is
   thrown away. Its five seeds are 2 DONE, 2 FAILED, 1 PAUSED, while the
   treatment arm's five are all DONE. The paired-by-seed comparison the
   hypothesis calls for cannot be computed from that either way, so the
   control arm has to be re-run regardless — fix the design first and the
   re-run is also the correct experiment.
3. Re-run this audit; the three INHERIT findings should disappear.

Not in the original plan; found 2026-09-20 by reading the re-synced form.

The canonical `experiment-default` document carries **24 of the 39 fields —
exactly the same 24 the JS mirror had**, missing exactly the same 15. This is
the same failure as Part 0, in a third copy of the schema: not the dashboard
mirror and not the form, but the seeded Mongo document.

`default_design_fields()` returns all 39, so the seeder is not the problem.
The guard above it is:

```python
SEED_VERSION = 1
existing = db.experiment_designs.find_one({"_id": DEFAULT_DESIGN_ID})
if existing and existing.get("version", 0) >= SEED_VERSION:
    return  # already current
```

The stored doc is `version: 1`, so every trainer start since the 15 fields
were added to `SCHEMA` has returned early. The bump the comment describes was
never made.

**Do this.** Set `SEED_VERSION = 2` in `_seed_canonical_experiment_design()`.
The upsert `$set`s `**default_design_fields()`, so the next trainer start
fills in the missing 15. The comment's worry about clobbering "users' edits
to Default" no longer applies — the tab renders Default read-only.

**Why it matters.** No run has been misconfigured by this: the canonical
values *are* the trainer defaults, so a missing key and a key set to its
default resolve identically in `apply_to_main_kwargs`. What breaks is
Default's stated job. It is documented as the reference every other design is
diffed against, the tab advertises it as "the trainer's canonical reference
set", and it currently under-reports itself as "24 overrides". Anything
cloned from it via **Duplicate** starts life missing the same 15 fields.

**Note on timing.** This only takes effect on the next trainer start. Do not
restart the trainer to hurry it — that would kill the running 100k-iteration
fly-brain job.

**You are done when.** After the next trainer restart,
`db.experiment_designs.findOne({_id:'experiment-default'})` has all 39 field
keys and the sidebar reads "39 overrides".

**Status — code change made 2026-09-20, still not in effect as of 2026-09-21
07:37 UTC.** `SEED_VERSION = 2` is committed to `robotaxi.py` and the file
compiles. The comment above the constant was also rewritten, since the old
"bump when defaults change in a meaningful way" wording is what made adding
fields look like it didn't need a bump.

Re-checked 2026-09-21. The stored doc is still `version: 1` with 24/39 fields,
and the reason is confirmed rather than assumed: the trainer process (PID 382)
started at 05:09:30 and `robotaxi.py` was last written at 07:06:38, so the
running interpreter holds a copy from before both this bump and Step 4's
publish function. Neither has ever executed.

One check worth recording, because the file contains two constants of the same
name: `SEED_VERSION = 1` at line 6689 is the *reward design* passthrough
seeder, a different function entirely. The experiment-design seeder is the one
at line 6762, and that is the one bumped.

**The reseed will not happen by itself when the job ends.** `robotaxi.py` is a
long-lived job loop, not a per-job process — it seeds at startup and then
polls forever. Finishing the fly-brain job returns it to polling; it does not
re-enter the seed path. Step 10 therefore needs a deliberate restart, not
patience.

**Cost of waiting.** Restarting before the job finishes throws its progress
away. Re-measured 2026-09-21 15:56 UTC: iteration 64,933 of 100,000, so about
7.5 hours remain.

Do not estimate this from the `elapsed_sec` in the TRAIN log lines. Those read
0.28 s/iter, implying 3.6 iterations/sec, but the sustained rate measured
across the 8.3 hours between iteration 27,602 and 64,933 is **1.25
iterations/sec** — evals, checkpoint writes and Unity resets are not in
`elapsed_sec`. An earlier estimate in this document used the per-iteration
figure and was optimistic by roughly a factor of three.

**One restart closes three open items:** this reseed, Step 4's publish
function (whose document currently exists only because I upserted it by hand),
and Step 4's "republished every boot" behaviour. Worth doing them together
rather than restarting twice.

**Alternative if 6.5 hours is too long:** apply the same `$set` by hand, as
Step 4 did. It is behaviourally inert — the 15 missing keys would be written
at exactly the values `apply_to_main_kwargs` already falls back to, so no run
resolves differently, and the tab renders Default read-only so there are no
edits to clobber. Leave `version` at 1 when doing this, so the seeder still
runs for real at the next restart instead of early-returning and leaving the
bump permanently unexercised.

**Decision 2026-09-21: wait.** Do not hand-apply, and do not restart early.
Job `6ab0bc6ac2736b990584404a` keeps running to 100,000 iterations; the
restart happens after it finishes, so the seeder and the publish function are
both exercised for real on their first run.

When the job reaches `DONE`, run:

```
# 1. confirm nothing else is mid-run before restarting
docker compose exec -T sim-controller tail -n 5 /tmp/trainer.log

# 2. rotate the log first. /tmp/trainer.log is a SYMLINK (currently to
#    trainer_resume6.log) and `tee` follows it and truncates the target,
#    so relaunching without this destroys the finished job's log.
docker compose exec -T sim-controller sh -c \
  'ln -sfn /tmp/trainer_resume7.log /tmp/trainer.log'

# 3. restart the trainer (this kills the job loop, not the container)
docker compose exec -d sim-controller \
  bash -c 'cd /python_ws/src && python -u robotaxi.py 2>&1 | tee /tmp/trainer.log'

# 4. Step 10: expect version 2 and 39/39 fields
docker compose exec -T dashboard node /dashboard/tools/audit_experiment_designs.js

# 5. Step 4: expect schema_registry.updated_at to move to the restart time,
#    proving the trainer published it rather than inheriting the hand-made
#    document that has sat there since 2026-09-21T07:11:19Z
```

Then check the sidebar reads "39 overrides" for Default, which is the
user-visible half of Step 10's done-condition.

---

## Known traps

1. **`server.ts` is not what runs.** The container executes
   `dashboard/build/server.js`; edit the TypeScript and rebuild, or the change
   is invisible.
2. **A schema-sync test would not run**, because this repo has no CI. Remove
   the duplication rather than testing it.
3. **The form's else-branch is a number input**, so any new non-`bool`,
   non-numeric type silently becomes a broken control. Part 4 fixes `json`;
   the same trap waits for `string` or `enum`. The read-only modals in
   `jobs.html` / `models.html` had the display half of this same problem
   (`String(val)` on an object); fixed in Step 2a, but a *third* code path
   means a new non-scalar type needs checking in three places.
4. **`/update_experiment_design` never deletes unknown fields.** Useful today
   — it is why nothing is corrupted — but it also means editing a design in
   the UI cannot clear a field the UI cannot see.
5. **Three tabs read this endpoint**, so verify `jobs.html` and `models.html`
   still work after Part 2, not just the Experiments tab.
