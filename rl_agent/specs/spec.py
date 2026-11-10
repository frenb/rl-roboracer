"""Robot-agnostic reward specs: a Markdown document that compiles per robot.

A spec says in prose what the robot should learn; a compiler (Mad Scientist's
``spec_compiler``) turns that prose into each robot's native reward design,
and the compiled code is kept in the same document:

    ---
    name: steady-progress
    kind: reward
    robots: [roboracer, microduck]
    microduck_task: Mjlab-Velocity-Flat-MicroDuck
    ---
    # Steady progress

    ## Goal
    ...
    ## Encourage / ## Discourage / ## Never / ## Success looks like
    ## Notes for roboracer   (optional, read only when compiling for that robot)

    ## Compiled: roboracer
    <!-- compiled inputs_sha=... code_sha=... at=... model=... -->
    ```python
    def reward_standard(...): ...
    ```

Everything before the first ``## Compiled:`` heading is the spec's source.
Each compiled section records the hash of the inputs it was compiled from and
of the code it produced, which gives every robot a status:

    missing  never compiled
    stale    the prose (or that robot's settings) changed since compiling
    edited   the code was changed by hand after compiling ("pinned")
    fresh    compiled from the current prose, untouched

Stdlib only: the dashboard runs this with a bare python3.
"""
import dataclasses
import datetime
import hashlib
import re

KIND = "reward"
ROBOTS = ("roboracer", "microduck")
DEFAULT_MICRODUCK_TASK = "Mjlab-Velocity-Flat-MicroDuck"
SECTIONS = ("Goal", "Encourage", "Discourage", "Never", "Success looks like")
# Frontmatter keys that change what a robot's code compiles from.
ROBOT_INPUT_KEYS = {"roboracer": (), "microduck": ("microduck_task",)}

_COMPILED_RE = re.compile(r"^##\s+Compiled:\s*(\S+)\s*$", re.M)
_META_RE = re.compile(r"<!--\s*compiled\s+(.*?)-->", re.S)
_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)^```\s*$", re.M | re.S)
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


class SpecError(ValueError):
    """The document is not a valid spec."""


@dataclasses.dataclass
class Compiled:
    code: str
    meta: dict

    @property
    def edited(self):
        return self.meta.get("code_sha") != sha(self.code)


@dataclasses.dataclass
class Spec:
    frontmatter: dict
    head: str            # frontmatter block + prose, verbatim
    prose: str           # prose without frontmatter, normalised
    compiled: dict       # robot -> Compiled

    @property
    def name(self):
        return self.frontmatter["name"]

    @property
    def robots(self):
        return list(self.frontmatter["robots"])

    def section(self, title):
        """The text under ``## <title>`` (case-insensitive), or ''."""
        return sections(self.prose).get(title.lower(), "")

    def inputs_sha(self, robot):
        extra = "".join(f"\n{k}={self.setting(robot, k)}" for k in ROBOT_INPUT_KEYS.get(robot, ()))
        return sha(self.prose + extra)

    def setting(self, robot, key):
        if key == "microduck_task":
            return self.frontmatter.get(key) or DEFAULT_MICRODUCK_TASK
        return self.frontmatter.get(key)

    def status(self, robot):
        c = self.compiled.get(robot)
        if c is None:
            return "missing"
        if c.meta.get("inputs_sha") != self.inputs_sha(robot):
            return "stale"
        return "edited" if c.edited else "fresh"

    def statuses(self):
        return {r: self.status(r) for r in self.robots}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalise(text):
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def _parse_value(raw):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return [v.strip().strip("'\"") for v in raw[1:-1].split(",") if v.strip()]
    return raw.strip("'\"")


def _frontmatter(text):
    if not text.startswith("---\n"):
        raise SpecError("a spec starts with a --- frontmatter block (name, kind, robots)")
    end = text.find("\n---", 4)
    if end < 0:
        raise SpecError("the frontmatter block is not closed with ---")
    fm = {}
    for n, line in enumerate(text[4:end].split("\n"), start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep or not key.strip().replace("_", "").isalnum():
            raise SpecError(f"frontmatter line {n}: expected 'key: value', got {line!r}")
        fm[key.strip()] = _parse_value(value)
    body_start = text.find("\n", end + 4)
    return fm, text[body_start + 1:] if body_start >= 0 else ""


def _check_frontmatter(fm):
    if not fm.get("name"):
        raise SpecError("frontmatter needs a name")
    kind = fm.setdefault("kind", KIND)
    if kind != KIND:
        raise SpecError(f"kind {kind!r} is not supported yet; only {KIND!r}")
    robots = fm.get("robots")
    if isinstance(robots, str):
        robots = fm["robots"] = [robots]
    if not robots:
        raise SpecError(f"frontmatter needs robots, e.g. robots: [{', '.join(ROBOTS)}]")
    unknown = [r for r in robots if r not in ROBOTS]
    if unknown:
        raise SpecError(f"unknown robots {unknown}; known: {', '.join(ROBOTS)}")


def sections(prose):
    """``{lowercased heading: body}`` for the prose's ``##`` sections."""
    out = {}
    matches = list(_SECTION_RE.finditer(prose))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prose)
        out[m.group(1).strip().lower()] = prose[m.end():end].strip()
    return out


def _compiled_sections(tail):
    out = {}
    matches = list(_COMPILED_RE.finditer(tail))
    for i, m in enumerate(matches):
        robot = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tail)
        body = tail[m.end():end]
        if robot not in ROBOTS:
            raise SpecError(f"'## Compiled: {robot}': unknown robot; known: {', '.join(ROBOTS)}")
        if robot in out:
            raise SpecError(f"'## Compiled: {robot}' appears twice")
        fence = _FENCE_RE.search(body)
        if not fence:
            raise SpecError(f"'## Compiled: {robot}' has no ```python code block")
        meta_m = _META_RE.search(body[:fence.start()])
        meta = {}
        if meta_m:
            for item in meta_m.group(1).split():
                k, _, v = item.partition("=")
                meta[k] = v
        out[robot] = Compiled(code=fence.group(1), meta=meta)
    return out


def parse(text):
    """Parse a spec document; raises ``SpecError``."""
    text = text.replace("\r\n", "\n")
    m = _COMPILED_RE.search(text)
    head, tail = (text[:m.start()], text[m.start():]) if m else (text, "")
    fm, body = _frontmatter(head)
    _check_frontmatter(fm)
    prose = _normalise(body)
    if "goal" not in sections(prose):
        raise SpecError("the spec needs a '## Goal' section saying what the robot should learn")
    return Spec(frontmatter=fm, head=head.rstrip() + "\n", prose=prose, compiled=_compiled_sections(tail))


def warnings(spec):
    """Advice that doesn't block saving."""
    found = sections(spec.prose)
    out = [f"no '## {s}' section" for s in SECTIONS[1:] if s.lower() not in found]
    for robot in spec.compiled:
        if robot not in spec.robots:
            out.append(f"compiled code for {robot} is ignored: it is not in robots")
    return out


def _meta_line(meta):
    return "<!-- compiled " + " ".join(f"{k}={v}" for k, v in meta.items()) + " -->"


def render(spec):
    """The document text: head, then a compiled section per robot that has one."""
    parts = [spec.head.rstrip() + "\n"]
    for robot in [r for r in ROBOTS if r in spec.compiled]:
        c = spec.compiled[robot]
        code = c.code if c.code.endswith("\n") else c.code + "\n"
        parts.append(f"\n## Compiled: {robot}\n{_meta_line(c.meta)}\n```python\n{code}```\n")
    return "".join(parts)


def set_compiled(text, robot, code, *, model="", at=None):
    """Replace ``robot``'s compiled section with freshly generated ``code``."""
    spec = parse(text)
    if robot not in spec.robots:
        raise SpecError(f"{robot} is not in this spec's robots")
    at = at or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"inputs_sha": spec.inputs_sha(robot), "code_sha": sha(code), "at": at}
    if model:
        meta["model"] = model
    spec.compiled[robot] = Compiled(code=code, meta=meta)
    return render(spec)


def repin(text, robot):
    """Accept ``robot``'s current code (hand-edited or not) for the current prose."""
    spec = parse(text)
    c = spec.compiled.get(robot)
    if c is None:
        raise SpecError(f"{robot} has no compiled code to keep")
    c.meta["inputs_sha"] = spec.inputs_sha(robot)
    return render(spec)


def edited_robots(old_text, new_text):
    """Robots whose compiled code ``new_text`` changes by hand relative to ``old_text``."""
    new = parse(new_text)
    try:
        old = parse(old_text).compiled if old_text else {}
    except SpecError:
        old = {}
    return sorted(r for r, c in new.compiled.items()
                  if c.edited and (r not in old or old[r].code != c.code))


SETTINGS_ID = "design_specs"


def strict_generated(db):
    """Whether hand-edited compiled code is refused (the dashboard's strict mode)."""
    doc = db.settings.find_one({"_id": SETTINGS_ID}) or {}
    return bool(doc.get("strict_generated"))


def unusable_reason(design, strict):
    """Why a reward design compiled from a spec can't train now, or None."""
    status = design.get("spec_status")
    if not design.get("spec_id") or status in (None, "fresh"):
        return None
    name = design.get("name") or design.get("_id")
    if status == "stale":
        return (f"reward design {name!r} is compiled from a reward spec whose prose changed since; "
                "recompile it (or keep the old code) on the Reward Specs tab")
    if status == "edited" and strict:
        return (f"reward design {name!r} has hand-edited code and strict mode is on; "
                "recompile it on the Reward Specs tab or turn strict mode off")
    return None


TEMPLATE = """---
name: steady-progress
kind: reward
robots: [roboracer, microduck]
microduck_task: Mjlab-Velocity-Flat-MicroDuck
---
# Steady progress

## Goal
Make progress toward the objective as fast as is safe: the next goal on the
course for the car, the commanded walking velocity for the duck.

## Encourage
- Progress toward the objective. This matters most.
- Staying pointed at the objective (heading for the car, velocity tracking for the duck).

## Discourage
- Jerky or wasteful control (a small penalty).

## Never
- Crashing or falling over: a large penalty.

## Success looks like
- roboracer: reaches goals quickly and finishes episodes without crashing.
- microduck: tracks the commanded velocity for a whole episode without falling.
"""
