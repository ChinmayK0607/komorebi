"""Author and admit a small structured-flow painting supervision set.

This file is deliberately a run-local candidate builder.  It authors sixteen
small p5.brush sketches (four single forms, four connected objects, four
spatial pairs, and four scenes), with a block-in and a separate improved
candidate for each.  It does not render on a workstation and it does not
decide whether a final candidate is visually better.  The ``render`` command
is Linux-only, binds both renderer source and asset-map hashes, and delegates
to the pinned renderer; ``export-sft`` accepts only source/hash-verified
receipts from that command.

The resulting 32 rows are deliberately small supervision candidates.  They
are not a claim that 32 examples are a sufficient training corpus, and a
render-valid final remains ``visual_review_pending`` until a human or an
explicit Flash decision reviews it.  An optional ``--include-finish`` export
adds eight terminal FINISHED rows only after those explicit decisions.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    # This is the exact text-brief prompt used by the native API-v2 rollout.
    # Importing the full native module is intentionally optional on the Mac:
    # its Verifiers dependency is supplied only by the Linux node.
    from painter.native_painting import (
        P5_BRUSH_API_REFERENCE_V2_SHA256,
        P5_BRUSH_API_REFERENCE_V2_VERSION,
        TEXT_INSPECT_CHOICE,
        TEXT_INSPECT_CHOICE_INITIAL,
        TEXT_SYSTEM_API_V2,
    )
    NATIVE_SYSTEM = TEXT_SYSTEM_API_V2
except ModuleNotFoundError as _native_import_error:  # Offline controller fallback.
    if not str(getattr(_native_import_error, "name", "")).startswith("verifiers"):
        raise
    from painter.contract import SYSTEM as _CONTRACT_SYSTEM

    _TEXT_SYSTEM = (
        _CONTRACT_SYSTEM
        .replace("Reproduce the supplied reference", "Depict the supplied PAINTING BRIEF")
        .replace("Inspect the reference and current canvas", "Inspect the PAINTING BRIEF and current canvas")
        .replace("faithfully matches the reference", "faithfully depicts the PAINTING BRIEF")
        .replace("matches the reference", "satisfies the PAINTING BRIEF")
        .replace(
            "Return a brief visual plan followed by a complete sketch inside a javascript code block.",
            "For a paint action, return one complete sketch inside a javascript code block and no prose plan.",
        )
        + "\nThe renderer has already loaded p5.js and p5.brush and exposes `brush` as a global. "
        + "For a paint action, output only JavaScript in a ```javascript code block; never output HTML, CSS, <script> tags, imports, or CDN URLs. "
        + "Do not declare, construct, or reassign `brush`; `new p5.brush(...)`, `p5.brush(...)`, and `brush.create(...)` are invalid here. "
        + "Do not redefine p5 globals such as translate or use brush.rect/brush.ellipse. Use brush.polygon for filled shapes and brush.line for strokes. "
        + "A valid skeleton is `function setup(){createCanvas(600,600,WEBGL);} "
        + "function draw(){translate(-300,-300); background('#f5efe4'); /* brush strokes */ noLoop();}`."
    )
    _P5_BRUSH_API_REFERENCE_V2 = """Pinned p5.brush drawing API reference v2 (opt-in; use only these verified calls):
`brush` is the already-loaded global singleton. Select a real named brush and its stroke color with
`brush.set(name, color, weight)`; `brush.box()` returns names such as `cpencil`, `pen`, `marker`, and `charcoal`.
`brush.stroke(color)` is the exported stroke-color setter. Exact drawing signatures are:
`brush.circle(x, y, radius, roughness?)`, `brush.rect(x, y, width, height, mode?)`,
`brush.line(x1, y1, x2, y2)`, and `brush.spline([[x, y, pressure], ...], curvature?)`.
Use `mode` `corner` or `center` for `brush.rect`; provide at least two points to `brush.spline`.
Do not invent global helpers named `circle`, `line`, `ellipse`, or `box`; call the methods on `brush`.
Do not define a second `setup()` or `draw()`. A minimal complete sketch is:
```javascript
function setup(){createCanvas(600,600,WEBGL);}
function draw(){translate(-width/2,-height/2);background('#f5efe4');brush.noFill();brush.noWash();brush.set('cpencil','#554d44',0.8);brush.stroke('#554d44');brush.line(80,80,520,520);noLoop();}
```
""".strip()
    P5_BRUSH_API_REFERENCE_V2_SHA256 = hashlib.sha256(_P5_BRUSH_API_REFERENCE_V2.encode("utf-8")).hexdigest()
    P5_BRUSH_API_REFERENCE_V2_VERSION = "p5-brush-api-v2"
    TEXT_INSPECT_CHOICE_INITIAL = (
        "No CURRENT CANVAS exists yet. Return one complete JavaScript sketch in one JavaScript fenced code block "
        "and no prose plan. FINISHED is invalid until a valid rendered canvas is available."
    )
    TEXT_INSPECT_CHOICE = (
        "Inspect the PAINTING BRIEF and CURRENT CANVAS. Choose exactly one response form:\n"
        "- If the observed canvas satisfies the brief, keep it and reply FINISHED without a code block.\n"
        "- Otherwise, revise it by returning one complete replacement JavaScript sketch in one JavaScript fenced "
        "code block. Do not append FINISHED to a code response."
    )
    NATIVE_SYSTEM = (
        _TEXT_SYSTEM.replace(
            "Do not redefine p5 globals such as translate or use brush.rect/brush.ellipse. Use brush.polygon for filled shapes and brush.line for strokes. ",
            "Do not redefine p5 globals such as translate. `brush.rect` is supported; `brush.ellipse` is not. Use brush.polygon for filled shapes and brush.line for strokes. ",
        )
        + "\n"
        + _P5_BRUSH_API_REFERENCE_V2
    )


RUN_ID = "structured-flow-rl-20261009"
SCHEMA = "painter.structured-flow-candidates.v1"
PROGRAM_SCHEMA = "painter.structured-flow-program.v1"
SFT_SCHEMA = "painter.structured-flow-sft.v1"
CANVAS_SIZE = 600
RENDER_TIMEOUT = 120
FLOW_INSTRUCTION = (
    "Paint the next version for the supplied brief. Return exactly one complete "
    "standalone JavaScript sketch inside one ```javascript code block and no "
    "prose outside that block. The target must use p5.brush and finish with "
    "noLoop()."
)
FLOW_HINT_FALLBACK = (
    "Plan before writing code: identify the subject hierarchy, placement, and "
    "largest shapes first; then make explicit canvas, palette, and brush "
    "choices. Revise by preserving correct structure and changing one control "
    "at a time. After observing the actual canvas, either return one meaningful "
    "complete painting revision or respond FINISHED when it satisfies the brief. "
    "For a paint action, return only executable painting code in the assistant turn."
)
FLOW_HOOK_MARKER = "[STRUCTURED_FLOW_POLICY]"


def _shared_flow_hint() -> str:
    """Load the launch agent's authoritative hint when the run is present."""
    launch_materialize = Path(__file__).with_name("materialize.py")
    if launch_materialize.is_file():
        spec = importlib.util.spec_from_file_location("structured_flow_launch_materialize", launch_materialize)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            value = getattr(module, "FLOW_HINT", None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return FLOW_HINT_FALLBACK


FLOW_HINT = _shared_flow_hint()


def flow_system() -> str:
    return NATIVE_SYSTEM + "\n" + FLOW_HOOK_MARKER + "\n" + FLOW_HINT
_HASH = re.compile(r"^[0-9a-f]{64}$")
_RAW_SHAPES = re.compile(
    r"(?<![.\w])(?:ellipse|rect|circle|line|triangle|quad|arc|point)\s*\("
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def native_prompt_parity() -> bool:
    """Assert the controller copy matches the native API-v2 prompt when present."""
    try:
        from painter.native_painting import (
            P5_BRUSH_API_REFERENCE_V2_SHA256 as native_api_sha,
            TEXT_SYSTEM_API_V2 as native_prompt,
        )
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "")).startswith("verifiers"):
            return False
        raise
    if native_prompt != NATIVE_SYSTEM or native_api_sha != P5_BRUSH_API_REFERENCE_V2_SHA256:
        raise RuntimeError("structured-flow prompt drifted from native TEXT_SYSTEM_API_V2")
    return True


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _local(origin: Sequence[int | float], points: Sequence[Sequence[int | float]]) -> list[list[int | float]]:
    """Expand object-local points for authoring-time validation and tests."""
    return [[origin[0] + p[0], origin[1] + p[1]] for p in points]


class Candidate:
    __slots__ = ("id", "category", "level", "brief", "palette", "stages", "split")

    def __init__(
        self,
        id: str,
        category: str,
        level: int,
        brief: str,
        palette: tuple[str, ...],
        stages: Mapping[str, str],
        split: str,
    ) -> None:
        self.id = id
        self.category = category
        self.level = level
        self.brief = brief
        self.palette = tuple(palette)
        self.stages = dict(stages)
        self.split = split


PRELUDE = r'''function setup(){createCanvas(600,600,WEBGL);pixelDensity(1);randomSeed(71);noiseSeed(71);}
function local(o,p){return p.map(function(q){return [o[0]+q[0],o[1]+q[1]];});}
function washShape(points,color,alpha,texture,bleed){
  brush.noStroke();brush.noWash();brush.fill(color,alpha);
  brush.fillBleed(bleed,"out");brush.fillTexture(texture,texture*0.5,false);brush.polygon(points);
}
function flatShape(points,color,alpha){
  brush.noStroke();brush.noWash();brush.fill(color,alpha);
  brush.fillBleed(0,"out");brush.fillTexture(0,0,false);brush.polygon(points);
}
function pencil(a,b,color,weight){
  brush.noFill();brush.noWash();brush.set("cpencil",color,weight);brush.line(a[0],a[1],b[0],b[1]);
}
function softLine(points,color,weight){
  brush.noFill();brush.noWash();brush.set("cpencil",color,weight);brush.spline(points,0.35);
}
'''


def _program(candidate: Candidate, improved: bool) -> str:
    """Build a complete target from the authored stage fragments."""
    stage_names = ("background", "silhouette", "majorvalues")
    if improved:
        stage_names += ("connections", "restraineddetails")
    chunks = [PRELUDE, "function draw(){translate(-300,-300);"]
    for stage in stage_names:
        chunks.append(f"// flow stage: {stage}")
        chunks.append(candidate.stages[stage].strip())
    chunks.append("noLoop();}\n")
    program = "\n".join(chunks)
    validate_program(program)
    return program


# Every stage is authored as a concrete sequence of broad brush marks.  The
# final stage is not a formatting cue: it changes visible geometry or value.
# ``local`` keeps each object's point list readable without introducing a
# second canvas transform or an affine frame.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "single-apple", "single_forms", 0,
        "A warm red apple sits slightly left of center on a pale tabletop; its stem and one small leaf lean toward the upper right.",
        ("#bd5b45", "#6e7c4c", "#55453f"),
        {
            "background": 'background("#f3eee4");flatShape([[80,475],[525,475],[525,550],[80,550]],"#e7ddcf",150);',
            "silhouette": 'flatShape([[164,394],[170,338],[202,302],[246,294],[285,307],[317,294],[361,306],[390,342],[394,394],[372,438],[337,470],[267,482],[208,463],[177,432]],"#bd5b45",235);washShape([[137,470],[415,470],[396,492],[150,492]],"#6e625b",55,0.05,0.008);',
            "majorvalues": 'washShape([[288,314],[341,316],[376,345],[378,405],[350,449],[315,466],[326,394]],"#713d3a",105,0.08,0.012);washShape([[185,339],[226,314],[264,318],[245,365],[207,389]],"#e28865",105,0.07,0.01);',
            "connections": 'pencil([274,312],[288,278],"#57443b",4);flatShape(local([288,278],[[0,0],[11,-8],[22,-5],[12,3]]),"#5c4738",220);flatShape(local([300,293],[[4,-2],[35,-16],[49,-8],[30,6]]),"#738651",200);',
            "restraineddetails": 'washShape([[224,337],[244,326],[260,331],[247,349],[229,355]],"#f2b28a",90,0.04,0.006);flatShape([[214,348],[228,339],[237,345],[227,354]],"#f8d3a0",125);pencil([222,353],[239,347],"#bb634c",0.7);',
        },
        "train",
    ),
    Candidate(
        "single-vase", "single_forms", 0,
        "A muted blue ceramic vase with a narrow neck and warm opening stands on a quiet cream surface, lit from the upper left.",
        ("#648b9a", "#d6b77e", "#4a5960"),
        {
            "background": 'background("#f2ede3");flatShape([[72,475],[528,475],[528,550],[72,550]],"#e4dacd",150);',
            "silhouette": 'flatShape([[230,246],[280,239],[320,246],[320,278],[342,307],[359,374],[350,445],[318,472],[238,472],[205,445],[196,374],[213,307],[230,278]],"#648b9a",235);washShape([[169,470],[385,470],[366,492],[183,492]],"#655e5b",58,0.05,0.008);flatShape([[230,245],[280,233],[320,243],[302,259],[247,260]],"#b99368",205);',
            "majorvalues": 'washShape([[287,270],[320,278],[339,330],[344,405],[321,453],[292,463],[305,383]],"#3f6676",112,0.08,0.012);washShape([[211,315],[236,274],[256,268],[248,380],[231,434]],"#a7c0c0",105,0.07,0.01);',
            "connections": 'flatShape([[229,238],[278,228],[323,240],[319,252],[279,245],[233,253]],"#d6b77e",210);flatShape([[242,243],[280,236],[311,244],[295,250],[253,249]],"#4a403c",190);',
            "restraineddetails": 'washShape([[239,292],[257,280],[268,285],[261,393],[247,415]],"#d2e0d6",78,0.04,0.006);pencil([215,438],[234,450],"#aac3c0",0.7);pencil([324,430],[334,420],"#315666",0.8);',
        },
        "train",
    ),
    Candidate(
        "single-orb", "single_forms", 0,
        "A muted blue-gray ceramic orb rests low in the frame, with one broad cool shadow and a restrained upper-left highlight.",
        ("#7c8fa1", "#b9c5ca", "#4f5964"),
        {
            "background": 'background("#f4efe6");flatShape([[70,478],[530,478],[530,550],[70,550]],"#e6ddd0",150);',
            "silhouette": 'flatShape([[173,397],[184,350],[220,313],[273,298],[327,307],[373,341],[392,392],[380,440],[341,474],[272,489],[211,469],[181,436]],"#7c8fa1",240);washShape([[142,474],[422,474],[402,498],[157,498]],"#666b72",62,0.04,0.008);',
            "majorvalues": 'washShape([[289,310],[337,323],[372,355],[375,415],[344,463],[310,478],[326,393]],"#465d72",120,0.08,0.012);washShape([[192,350],[224,317],[263,309],[249,380],[214,416]],"#b9c5ca",110,0.07,0.01);',
            "connections": 'flatShape([[180,435],[210,452],[272,476],[254,487],[208,469]],"#596775",105);pencil([211,350],[238,329],"#d7e0df",0.9);pencil([334,340],[359,367],"#405668",0.9);',
            "restraineddetails": 'flatShape([[214,345],[234,328],[252,337],[238,357]],"#e7eee6",115);pencil([253,443],[272,452],"#d5e0df",0.75);flatShape([[302,469],[332,460],[339,470],[315,481]],"#3f5364",72);',
        },
        "train",
    ),
    Candidate(
        "single-lantern", "single_forms", 0,
        "A small paper lantern hangs in the middle of a pale wall, with a dark cap, a warm inner glow, and a short cord.",
        ("#d77d49", "#eed49b", "#4b4a49"),
        {
            "background": 'background("#eee9df");flatShape([[78,474],[522,474],[522,550],[78,550]],"#ded6ca",140);',
            "silhouette": 'pencil([300,94],[300,129],"#4b4a49",2.3);flatShape([[276,126],[324,126],[339,143],[330,158],[270,158],[261,143]],"#4b4a49",225);flatShape([[252,158],[348,158],[369,207],[366,373],[343,430],[257,430],[234,373],[231,207]],"#d77d49",235);washShape([[214,428],[382,428],[368,450],[225,450]],"#71665d",54,0.04,0.006);',
            "majorvalues": 'washShape([[284,170],[330,166],[351,214],[350,358],[327,409],[307,419],[316,300]],"#ad503d",118,0.08,0.012);washShape([[245,205],[264,173],[280,180],[275,386],[255,408]],"#edb369",105,0.07,0.01);',
            "connections": 'pencil([300,429],[300,474],"#4b4a49",2);pencil([247,195],[247,369],"#efc57d",1.1);pencil([353,195],[353,369],"#74433d",1.1);',
            "restraineddetails": 'flatShape([[261,224],[278,214],[278,365],[260,381]],"#f1cb7f",90);flatShape([[325,213],[343,222],[343,377],[326,365]],"#743d39",76);pencil([268,152],[332,152],"#d7c0a3",0.8);',
        },
        "train",
    ),

    Candidate(
        "connected-cup-saucer", "connected_objects", 1,
        "A blue cup sits on a cream saucer; its handle opens to the right and a small spoon rests behind the rim.",
        ("#557d93", "#ded0af", "#4d514d"),
        {
            "background": 'background("#f3eee5");flatShape([[65,470],[535,470],[535,550],[65,550]],"#e3d9cb",145);',
            "silhouette": 'flatShape([[150,436],[449,436],[470,458],[443,480],[157,480],[130,458]],"#ded0af",220);flatShape([[207,292],[351,292],[370,340],[354,430],[222,430],[188,340]],"#557d93",235);washShape([[178,429],[389,429],[374,450],[195,450]],"#70675e",58,0.05,0.008);',
            "majorvalues": 'washShape([[303,302],[351,309],[363,350],[348,413],[315,426],[319,365]],"#315c73",118,0.08,0.012);washShape([[202,320],[230,300],[251,307],[245,405],[224,416]],"#9db6bd",105,0.07,0.01);flatShape([[204,289],[278,273],[354,291],[345,313],[214,310]],"#e3d4b5",220);flatShape([[222,290],[277,282],[337,293],[323,304],[232,303]],"#5b4840",210);',
            "connections": 'flatShape(local([347,343],[[0,-24],[44,-14],[61,14],[48,46],[27,59],[12,45],[31,32],[37,11],[28,0],[0,0]]),"#557d93",215);pencil([382,302],[410,251],"#4d514d",1.8);pencil([410,251],[425,244],"#4d514d",1.8);',
            "restraineddetails": 'flatShape([[225,325],[243,317],[244,397],[229,405]],"#d6edf0",90);pencil([157,458],[442,458],"#9e8e70",0.8);pencil([268,280],[302,278],"#f2e5c9",0.7);',
        },
        "train",
    ),
    Candidate(
        "connected-bird-branch", "connected_objects", 1,
        "A russet songbird perches on a green branch, with its tail pointing down-left and a tiny berry cluster nearby.",
        ("#a95d48", "#75834f", "#4f423f"),
        {
            "background": 'background("#e9eee8");flatShape([[72,469],[528,469],[528,550],[72,550]],"#dce4d8",120);',
            "silhouette": 'flatShape([[140,423],[191,409],[238,362],[294,330],[355,337],[397,372],[400,415],[372,448],[313,454],[260,438],[211,444]],"#a95d48",235);pencil([105,454],[474,451],"#75834f",9);flatShape([[206,425],[163,469],[136,472],[176,432]],"#a95d48",215);',
            "majorvalues": 'flatShape([[292,334],[355,340],[393,373],[362,407],[306,391]],"#7d443b",145);washShape([[186,409],[234,364],[270,355],[261,427],[214,438]],"#d78962",105,0.08,0.012);flatShape([[347,382],[395,414],[372,447],[342,430]],"#713c39",100);',
            "connections": 'pencil([247,439],[244,469],"#4f423f",2.2);pencil([278,442],[286,468],"#4f423f",2.2);flatShape([[350,348],[365,337],[377,350],[360,360]],"#e1a060",170);pencil([398,398],[418,405],"#4f423f",1.7);',
            "restraineddetails": 'flatShape([[347,365],[359,359],[366,370],[355,381]],"#eee0a1",175);flatShape([[175,435],[188,430],[190,443],[178,450]],"#9ca760",175);flatShape([[460,422],[470,413],[480,423],[469,433]],"#a54f49",180);flatShape([[476,432],[486,423],[496,433],[485,442]],"#a54f49",170);',
        },
        "train",
    ),
    Candidate(
        "connected-boat-sail", "connected_objects", 1,
        "A small ochre sailboat crosses a quiet blue water band; its triangular sail is tied to one dark mast.",
        ("#c5864b", "#d6d9cc", "#446979"),
        {
            "background": 'background("#e9eee9");flatShape([[70,404],[530,404],[530,550],[70,550]],"#a9c4c3",150);',
            "silhouette": 'flatShape([[136,390],[463,390],[430,445],[175,445]],"#8a583e",235);flatShape([[281,112],[282,394],[292,394],[292,112]],"#4c4b47",220);flatShape([[281,132],[157,348],[281,348]],"#d6d9cc",230);washShape([[111,442],[467,442],[447,463],[128,463]],"#54747b",58,0.05,0.008);',
            "majorvalues": 'flatShape([[281,142],[269,344],[164,344]],"#f0e3bb",115);flatShape([[289,135],[291,346],[274,346]],"#b4c1b8",105);washShape([[176,391],[429,391],[414,428],[200,428]],"#c09058",110,0.07,0.01);',
            "connections": 'pencil([281,131],[281,348],"#4c4b47",1.7);pencil([168,349],[407,349],"#6d6b5b",1.1);pencil([300,344],[391,388],"#4c4b47",1);pencil([200,388],[281,347],"#4c4b47",1);',
            "restraineddetails": 'flatShape([[201,397],[235,397],[229,426],[211,426]],"#5c7680",135);flatShape([[333,397],[367,397],[365,425],[343,425]],"#5c7680",135);pencil([154,456],[216,458],"#d3e1dc",1.2);pencil([386,470],[447,470],"#d3e1dc",1.1);',
        },
        "train",
    ),
    Candidate(
        "connected-teapot", "connected_objects", 1,
        "A small copper teapot sits on a warm paper ground; its lid, spout, and open handle visibly attach to the rounded body.",
        ("#b8734e", "#d7ae70", "#4d5555"),
        {
            "background": 'background("#f3ede1");flatShape([[70,468],[530,468],[530,550],[70,550]],"#e4d8c8",145);',
            "silhouette": 'flatShape([[202,306],[233,271],[294,257],[356,272],[392,310],[399,390],[367,439],[239,439],[201,390]],"#b8734e",235);flatShape([[256,269],[256,237],[281,216],[329,216],[350,237],[350,269]],"#b8734e",225);flatShape([[280,216],[329,216],[340,229],[270,229]],"#d7ae70",210);flatShape([[202,313],[153,292],[112,312],[151,335],[207,350]],"#b8734e",220);washShape([[177,437],[420,437],[397,463],[196,463]],"#6d6257",54,0.04,0.006);',
            "majorvalues": 'washShape([[313,276],[373,296],[386,345],[378,395],[345,428],[327,427],[339,355]],"#824c3c",118,0.08,0.012);washShape([[211,320],[238,282],[265,279],[259,401],[232,421]],"#d79a66",105,0.07,0.01);flatShape([[275,239],[329,239],[335,253],[268,253]],"#8f543c",130);',
            "connections": 'flatShape(local([351,309],[[0,0],[46,8],[69,30],[67,48],[48,52],[36,38],[49,31],[45,22],[0,17]]),"#b8734e",215);pencil([202,323],[151,311],"#4d5555",1.5);pencil([151,311],[116,329],"#4d5555",1.5);flatShape([[107,324],[123,314],[130,329],[117,339]],"#d7ae70",180);',
            "restraineddetails": 'flatShape([[235,316],[253,303],[258,394],[242,408]],"#e9c18b",92);flatShape([[289,294],[307,288],[309,405],[291,412]],"#df965e",78);pencil([271,230],[330,230],"#e8cc97",0.8);pencil([326,271],[350,279],"#e0bd85",0.7);',
        },
        "train",
    ),

    Candidate(
        "pair-apple-pear", "spatial_pairs", 2,
        "A red apple sits on the left and a smaller golden pear on the right, separated by open tabletop space and sharing one soft ground plane.",
        ("#b95743", "#c8a04b", "#68744d"),
        {
            "background": 'background("#f3eee5");flatShape([[62,470],[538,470],[538,550],[62,550]],"#e4d9cb",145);',
            "silhouette": 'flatShape([[108,411],[120,362],[149,334],[185,330],[213,348],[226,390],[218,431],[187,456],[143,455]],"#b95743",230);flatShape([[335,427],[339,373],[360,339],[392,316],[423,323],[443,357],[442,417],[420,454],[371,458]],"#c8a04b",230);washShape([[88,456],[242,456],[232,474],[97,474]],"#70655c",46,0.04,0.006);washShape([[320,458],[464,458],[449,477],[330,477]],"#70655c",46,0.04,0.006);',
            "majorvalues": 'washShape([[170,337],[211,352],[214,409],[181,446],[178,391]],"#713d3a",105,0.08,0.012);washShape([[389,324],[425,335],[431,404],[409,446],[405,380]],"#9e7138",105,0.07,0.01);flatShape([[145,341],[168,330],[178,349],[157,360]],"#e28362",75);',
            "connections": 'pencil([171,337],[181,311],"#57453b",2.7);flatShape(local([181,311],[[0,0],[12,-7],[22,-4],[11,5]]),"#65794a",185);pencil([393,326],[388,294],"#57453b",2.4);flatShape(local([388,294],[[0,0],[27,-9],[36,-3],[19,6]]),"#75814c",175);',
            "restraineddetails": 'flatShape([[142,357],[155,349],[165,357],[157,368]],"#f2ba8b",95);flatShape([[365,345],[383,334],[392,345],[380,355]],"#f1d18a",92);pencil([257,460],[299,460],"#c8bba6",0.6);',
        },
        "train",
    ),
    Candidate(
        "pair-book-mug", "spatial_pairs", 2,
        "A closed blue book lies at lower left and a warm mug stands at upper right, with a clear diagonal relationship across the desk.",
        ("#4f708a", "#ba7450", "#dfc89d"),
        {
            "background": 'background("#f1ece2");flatShape([[60,470],[540,470],[540,550],[60,550]],"#e0d5c6",145);',
            "silhouette": 'flatShape([[103,369],[273,337],[321,390],[145,427]],"#4f708a",235);flatShape([[355,260],[445,260],[461,290],[449,397],[370,397],[346,290]],"#ba7450",230);washShape([[337,397],[467,397],[455,416],[346,416]],"#75675c",50,0.04,0.006);',
            "majorvalues": 'flatShape([[104,369],[273,337],[299,373],[129,406]],"#7192a3",115);flatShape([[129,406],[299,373],[321,390],[145,427]],"#37556c",150);washShape([[416,262],[445,270],[453,300],[441,385],[420,392],[425,322]],"#824c42",110,0.08,0.012);flatShape([[355,259],[401,247],[445,260],[438,278],[362,274]],"#dfc89d",215);flatShape([[365,259],[400,252],[434,263],[426,271],[372,268]],"#4e403c",205);',
            "connections": 'pencil([142,389],[290,354],"#d3c5a4",0.8);pencil([378,302],[353,295],"#ba7450",2);pencil([444,307],[469,319],"#ba7450",2);pencil([469,319],[474,345],"#ba7450",2);pencil([474,345],[457,358],"#ba7450",2);',
            "restraineddetails": 'flatShape([[167,372],[261,354],[275,368],[181,388]],"#e2d8bf",90);pencil([186,402],[271,384],"#334d61",0.65);flatShape([[381,282],[395,277],[395,374],[383,379]],"#e3ae7a",85);',
        },
        "train",
    ),
    Candidate(
        "pair-rock-flower", "spatial_pairs", 2,
        "A cool gray rock anchors the left foreground while a yellow flower on a green stem rises on the right, leaving a readable gap between them.",
        ("#7b7770", "#d0a64a", "#6e8052"),
        {
            "background": 'background("#edf0e6");flatShape([[62,472],[538,472],[538,550],[62,550]],"#dbe3d1",145);',
            "silhouette": 'flatShape([[99,439],[122,391],[165,369],[221,376],[253,410],[246,447],[207,468],[142,467]],"#7b7770",235);pencil([382,465],[375,330],"#6e8052",5.5);flatShape([[340,339],[355,301],[382,281],[414,289],[438,321],[430,350],[394,366],[360,356]],"#d0a64a",210);washShape([[79,455],[264,455],[248,478],[95,478]],"#6d665e",53,0.04,0.006);',
            "majorvalues": 'flatShape([[122,391],[165,369],[190,420],[142,454],[101,437]],"#a09a87",210);flatShape([[190,420],[221,378],[249,410],[240,443],[205,463]],"#554f4d",190);washShape([[367,310],[408,292],[431,322],[416,346],[386,354]],"#e7c866",110,0.07,0.01);',
            "connections": 'flatShape(local([375,330],[[0,0],[15,-6],[23,5],[16,12],[2,8]]),"#6e8052",200);pencil([382,351],[382,465],"#536b45",1.3);flatShape([[369,458],[396,458],[414,470],[355,470]],"#6e8052",100);',
            "restraineddetails": 'flatShape([[376,311],[390,300],[402,311],[391,323]],"#f4e5a0",120);pencil([135,409],[171,391],"#c3b18f",0.8);flatShape([[387,456],[396,447],[403,456],[397,465]],"#d8c56c",100);',
        },
        "train",
    ),
    Candidate(
        "pair-sun-cloud", "spatial_pairs", 2,
        "A pale sun occupies the upper left and a long blue cloud drifts across the upper right above a quiet lavender horizon.",
        ("#e0b45e", "#a6bfca", "#7b7890"),
        {
            "background": 'background("#f1ecdf");flatShape([[60,420],[540,420],[540,550],[60,550]],"#cbc3d0",145);',
            "silhouette": 'flatShape([[147,183],[183,151],[229,143],[270,166],[276,212],[246,246],[194,251],[154,226]],"#e0b45e",230);flatShape([[301,236],[319,203],[355,197],[382,213],[420,203],[458,220],[477,258],[462,289],[317,289],[294,267]],"#a6bfca",220);flatShape([[60,403],[540,403],[540,430],[60,430]],"#7b7890",160);',
            "majorvalues": 'washShape([[183,152],[224,148],[261,176],[257,215],[224,237],[202,199]],"#f0d37e",100,0.07,0.01);washShape([[355,199],[382,214],[419,204],[460,224],[449,264],[404,279],[374,249]],"#7f9da9",110,0.08,0.012);flatShape([[61,404],[539,404],[539,416],[61,416]],"#a49bb1",145);',
            "connections": 'for(let x=104;x<285;x+=28){pencil([x,393],[x+9,380],"#978da0",0.9);}pencil([301,289],[464,289],"#718d99",1.1);pencil([319,267],[343,267],"#e0eff0",1);',
            "restraineddetails": 'flatShape([[189,170],[204,157],[220,168],[209,181]],"#f7e6a9",100);flatShape([[333,227],[348,214],[365,224],[353,238]],"#d9e6e5",90);pencil([116,428],[186,428],"#aba1b1",0.8);',
        },
        "train",
    ),

    Candidate(
        "scene-hill-cabin", "simple_scenes", 3,
        "A tiny ochre cabin sits on a green hill beneath a cool sky, with one tree at the left and a winding path toward the door.",
        ("#c48b52", "#71855d", "#7898a3"),
        {
            "background": 'background("#dfe9e5");flatShape([[60,60],[540,60],[540,410],[60,410]],"#b9d0d0",135);flatShape([[60,410],[540,410],[540,550],[60,550]],"#8aa077",150);',
            "silhouette": 'flatShape([[214,302],[298,244],[383,302],[383,415],[214,415]],"#c48b52",230);flatShape([[201,305],[299,225],[396,305],[379,319],[299,254],[219,319]],"#805745",225);flatShape([[111,391],[142,315],[180,276],[202,317],[207,404]],"#71855d",220);flatShape([[74,424],[238,424],[321,401],[485,417],[540,450],[60,450]],"#71855d",210);',
            "majorvalues": 'flatShape([[215,304],[298,250],[298,414],[215,414]],"#d6a261",110);washShape([[299,255],[378,306],[378,414],[323,414],[326,340]],"#8f5f47",115,0.08,0.012);flatShape([[260,335],[329,335],[329,415],[260,415]],"#76503f",185);flatShape([[271,352],[302,352],[302,388],[271,388]],"#e3bc70",155);washShape([[98,448],[510,438],[540,470],[60,482]],"#5d7957",75,0.07,0.01);',
            "connections": 'pencil([299,251],[299,412],"#725044",1);pencil([109,392],[82,418],"#71855d",5);pencil([137,386],[178,369],"#71855d",4);flatShape([[300,414],[326,414],[352,465],[314,466]],"#c6a876",150);',
            "restraineddetails": 'flatShape([[234,327],[249,318],[249,349],[234,354]],"#e0d8b4",100);flatShape([[343,326],[357,332],[357,356],[343,351]],"#e0d8b4",95);pencil([403,394],[444,390],"#b7c8ba",1);flatShape([[171,293],[186,285],[194,296],[181,306]],"#98aa71",110);',
        },
        "dev",
    ),
    Candidate(
        "scene-pond-reeds", "simple_scenes", 3,
        "A low blue pond fills the foreground, with three reeds at the right, a small stepping stone, and a peach evening bank.",
        ("#739ba1", "#718055", "#d49b77"),
        {
            "background": 'background("#f0e5dc");flatShape([[60,70],[540,70],[540,370],[60,370]],"#e3bca7",135);flatShape([[60,370],[540,370],[540,550],[60,550]],"#739ba1",150);',
            "silhouette": 'flatShape([[60,351],[180,337],[300,351],[410,330],[540,345],[540,386],[60,386]],"#9d806a",170);flatShape([[374,440],[407,423],[440,438],[432,458],[389,463]],"#7b7970",220);pencil([450,443],[456,288],"#718055",4.3);pencil([474,445],[502,300],"#718055",3.4);pencil([426,444],[405,311],"#718055",3.5);',
            "majorvalues": 'washShape([[60,389],[540,389],[540,550],[60,550]],"#6c929d",105,0.08,0.012);flatShape([[60,371],[540,371],[540,388],[60,388]],"#d49b77",130);washShape([[384,425],[418,413],[443,436],[430,452],[394,452]],"#a49b86",95,0.07,0.01);',
            "connections": 'pencil([456,288],[443,321],"#718055",1.1);pencil([502,300],[472,331],"#718055",1);pencil([405,311],[434,344],"#718055",1);flatShape([[452,282],[464,279],[464,294],[454,296]],"#d4a06e",110);',
            "restraineddetails": 'for(let x=112;x<500;x+=54){pencil([x,430],[x+30,430],"#c0d5ce",0.9);}flatShape([[455,341],[462,334],[469,341],[463,348]],"#d5c88d",100);flatShape([[491,355],[498,348],[505,355],[499,362]],"#d5c88d",90);',
        },
        "dev",
    ),
    Candidate(
        "scene-window-plant", "simple_scenes", 3,
        "A terracotta plant sits on an interior sill before a blue window, with three leaves reaching into the light.",
        ("#bf714f", "#708856", "#7599a7"),
        {
            "background": 'background("#eee8dd");flatShape([[60,60],[540,60],[540,438],[60,438]],"#7599a7",145);flatShape([[60,438],[540,438],[540,550],[60,550]],"#d7c5ad",150);',
            "silhouette": 'flatShape([[117,93],[483,93],[483,422],[117,422]],"#7599a7",210);flatShape([[87,410],[513,410],[513,450],[87,450]],"#8c7664",205);flatShape([[250,365],[350,365],[339,463],[261,463]],"#bf714f",230);pencil([299,368],[299,231],"#708856",4.2);flatShape([[298,266],[252,229],[226,231],[249,262],[292,282]],"#708856",210);flatShape([[299,301],[349,262],[376,270],[354,299],[308,320]],"#708856",205);',
            "majorvalues": 'flatShape([[117,94],[290,94],[290,409],[117,409]],"#9fc0c4",110);washShape([[290,94],[482,94],[482,409],[304,409]],"#4f788c",95,0.08,0.012);washShape([[263,370],[338,370],[331,451],[270,451]],"#994f43",110,0.07,0.01);',
            "connections": 'pencil([299,230],[299,366],"#516844",1.4);pencil([299,281],[252,241],"#516844",1.1);pencil([304,318],[359,276],"#516844",1.1);flatShape([[251,444],[352,444],[346,462],[258,462]],"#d49067",70);',
            "restraineddetails": 'flatShape([[236,231],[249,223],[263,236],[250,246]],"#b9d18e",90);flatShape([[354,268],[369,263],[378,275],[362,283]],"#b9d18e",90);pencil([140,116],[272,116],"#d7e4dd",0.9);pencil([324,116],[463,116],"#d7e4dd",0.9);',
        },
        "dev",
    ),
    Candidate(
        "scene-market-stall", "simple_scenes", 3,
        "A striped market stall spans the center, with a wooden counter, two fruit crates, and a small bunting line against a pale street.",
        ("#d28b55", "#7b8b66", "#5a6871"),
        {
            "background": 'background("#f0ebe1");flatShape([[60,75],[540,75],[540,460],[60,460]],"#e4ded2",140);flatShape([[60,460],[540,460],[540,550],[60,550]],"#d1c2af",145);',
            "silhouette": 'flatShape([[141,202],[459,202],[459,424],[141,424]],"#7b8b66",215);flatShape([[114,192],[487,192],[468,264],[132,264]],"#d28b55",230);flatShape([[156,264],[444,264],[444,410],[156,410]],"#8b6448",200);flatShape([[146,394],[454,394],[454,427],[146,427]],"#5a6871",220);',
            "majorvalues": 'for(let x=132;x<468;x+=56){flatShape([[x,194],[x+30,194],[x+18,262],[x-2,262]],"#f1d9a2",145);}flatShape([[171,324],[254,324],[254,393],[171,393]],"#b7744d",205);flatShape([[337,324],[421,324],[421,393],[337,393]],"#b7744d",205);washShape([[257,269],[443,269],[443,390],[257,390]],"#6e4d3c",78,0.07,0.01);',
            "connections": 'pencil([145,194],[145,425],"#5a6871",2.5);pencil([455,194],[455,425],"#5a6871",2.5);pencil([168,298],[431,298],"#dec69f",1.1);pencil([194,184],[214,151],"#5a6871",1);pencil([214,151],[244,174],"#5a6871",1);pencil([244,174],[274,146],"#5a6871",1);',
            "restraineddetails": 'flatShape([[190,349],[205,337],[220,349],[206,362]],"#cc423d",180);flatShape([[220,370],[234,358],[248,370],[235,383]],"#dfb34d",180);flatShape([[356,349],[371,337],[386,349],[371,362]],"#7c9d55",180);flatShape([[390,370],[405,358],[418,370],[405,383]],"#d5a641",180);pencil([280,340],[321,340],"#c08e63",0.8);',
        },
        "dev",
    ),
)


def validate_program(program: str) -> None:
    """Validate authored source without invoking a JavaScript renderer."""
    if not isinstance(program, str) or not program.strip():
        raise ValueError("program must be non-empty text")
    if program.count("function setup()") != 1 or program.count("function draw()") != 1:
        raise ValueError("program must define exactly one setup and draw")
    translations = re.findall(r"\btranslate\s*\(([^)]*)\)", program)
    if len(translations) != 1 or re.sub(r"\s", "", translations[0]) != "-300,-300":
        raise ValueError("program must use exactly one translate(-300,-300)")
    if "brush.polygon" not in program or "brush.line" not in program:
        raise ValueError("program must use p5.brush polygon and line marks")
    if _RAW_SHAPES.search(program):
        raise ValueError("raw p5 shape drawing is not an authored target")
    if "background(" not in program or "noLoop();}" not in program:
        raise ValueError("program must include background and finish draw with noLoop")
    if not all(f"// flow stage: {stage}" in program for stage in ("background", "silhouette", "majorvalues")):
        raise ValueError("program is missing the required painting flow")
    if "brush.fillTexture(texture,texture*0.5,false)" not in program or "brush.fillBleed(bleed,\"out\")" not in program:
        raise ValueError("program must make low texture and bleed controls explicit")


def _candidate_by_id(candidate_id: str) -> Candidate:
    for candidate in CANDIDATES:
        if candidate.id == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def source_records(output: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        folder = output / "programs" / candidate.id
        for version, improved in (("block_in", False), ("improved_final", True)):
            path = folder / f"{version}.js"
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append({
                "candidate_id": candidate.id,
                "category": candidate.category,
                "version": version,
                "path": path.relative_to(output).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "standalone": True,
            })
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def build(output: str | Path) -> dict[str, Any]:
    """Write authored sources and hash-bound candidate/prompt manifests."""
    prompt_parity = native_prompt_parity()
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for candidate in CANDIDATES:
        folder = root / "programs" / candidate.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "block_in.js").write_text(_program(candidate, False), encoding="utf-8")
        (folder / "improved_final.js").write_text(_program(candidate, True), encoding="utf-8")

    records = source_records(root)
    source_manifest = {
        "schema": PROGRAM_SCHEMA,
        "run_id": RUN_ID,
        "source_status": "candidate_authored_unrendered",
        "quality_claim": False,
        "native_system_sha256": sha256_text(NATIVE_SYSTEM),
        "sources": records,
    }
    source_manifest_bytes = canonical_json(source_manifest)
    source_manifest_sha = sha256_bytes(source_manifest_bytes)
    prompt_manifest = {
        "schema": "painter.prompt-manifest.v1",
        "run_id": RUN_ID,
        "native_system_sha256": sha256_text(NATIVE_SYSTEM),
        "native_system": NATIVE_SYSTEM,
        "native_api_reference_version": P5_BRUSH_API_REFERENCE_V2_VERSION,
        "native_api_reference_sha256": P5_BRUSH_API_REFERENCE_V2_SHA256,
        "native_prompt_parity_available": prompt_parity,
        "flow_hook_marker": FLOW_HOOK_MARKER,
        "flow_hint": FLOW_HINT,
        "flow_hint_sha256": sha256_text(FLOW_HINT),
        "flow_system_sha256": sha256_text(flow_system()),
        "flow_instruction": FLOW_INSTRUCTION,
        "flow_instruction_sha256": sha256_text(FLOW_INSTRUCTION),
        "target_format": "one_javascript_fence_no_prose",
    }
    prompt_manifest_sha = sha256_bytes(canonical_json(prompt_manifest))
    _write_json(root / "source-manifest.json", source_manifest)
    _write_json(root / "prompt-manifest.json", prompt_manifest)

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(CANDIDATES):
        row_records = [r for r in records if r["candidate_id"] == candidate.id]
        by_version = {r["version"]: r for r in row_records}
        rows.append({
            "id": candidate.id,
            "source_group": candidate.id,
            "category": candidate.category,
            "level": candidate.level,
            "split": candidate.split,
            "brief": candidate.brief,
            "palette": list(candidate.palette),
            "flow": ["background", "silhouette", "majorvalues", "connections", "restraineddetails"],
            "programs": [by_version["block_in"]["path"], by_version["improved_final"]["path"]],
            "source_sha256": {v: by_version[v]["sha256"] for v in ("block_in", "improved_final")},
            "reference": f"renders/{candidate.id}/improved_final.png",
            "validity": "awaiting_linux_render",
            "quality_status": "visual_review_pending",
            "improved_reviewed": False,
            "admission": "candidate_only",
            "quality_claim": False,
            "generalisation_scope": "sixteen authored candidates; no claim beyond these source groups",
            "curriculum_index": index,
        })
    manifest = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "count": len(rows),
        "program_count": len(records),
        "row_count_after_pairing": len(rows) * 2,
        "source_manifest_sha256": source_manifest_sha,
        "prompt_manifest_sha256": prompt_manifest_sha,
        "native_system_sha256": sha256_text(NATIVE_SYSTEM),
        "rows": rows,
    }
    _write_json(root / "manifest.json", manifest)
    (root / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return manifest


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"candidate manifest is missing: {path}; run build first")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("run_id") != RUN_ID:
        raise ValueError("manifest is not a structured-flow candidate manifest")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(CANDIDATES):
        raise ValueError("manifest must contain exactly sixteen candidates")
    return value


def _verify_sources(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_path = root / "source-manifest.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    if sha256_bytes(canonical_json(source_manifest)) != manifest.get("source_manifest_sha256"):
        raise ValueError("source manifest SHA-256 mismatch")
    records = source_manifest.get("sources")
    if not isinstance(records, list) or len(records) != 32:
        raise ValueError("source manifest must contain thirty-two programs")
    for record in records:
        path = root / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = file_sha256(path)
        if digest != record.get("sha256"):
            raise ValueError(f"source SHA-256 mismatch for {path}")
        validate_program(path.read_text(encoding="utf-8"))
    return records


def _verify_prompt(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    prompt_path = root / "prompt-manifest.json"
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    expected = manifest.get("prompt_manifest_sha256")
    if not isinstance(expected, str) or sha256_bytes(canonical_json(prompt)) != expected:
        raise ValueError("prompt manifest SHA-256 mismatch")
    if prompt.get("native_system_sha256") != sha256_text(NATIVE_SYSTEM):
        raise ValueError("prompt manifest native system hash mismatch")
    if prompt.get("native_api_reference_version") != P5_BRUSH_API_REFERENCE_V2_VERSION:
        raise ValueError("prompt manifest is not bound to native p5.brush API v2")
    if prompt.get("native_api_reference_sha256") != P5_BRUSH_API_REFERENCE_V2_SHA256:
        raise ValueError("prompt manifest p5.brush API reference hash mismatch")
    if prompt.get("flow_hint_sha256") != sha256_text(FLOW_HINT):
        raise ValueError("prompt manifest flow hint hash mismatch")
    if prompt.get("flow_hook_marker") != FLOW_HOOK_MARKER or prompt.get("flow_system_sha256") != sha256_text(flow_system()):
        raise ValueError("prompt manifest structured-flow system hook mismatch")
    native_prompt_parity()
    return prompt


def _renderer_result(source: Path, png: Path, renderer: Path, renderer_python: Path, run_as_user: str) -> dict[str, Any]:
    png.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/sbin/runuser", "-u", run_as_user, "--", str(renderer_python), str(renderer),
        str(source), "--output", str(png), "--backend", "swiftshader", "--timeout", str(RENDER_TIMEOUT),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=RENDER_TIMEOUT + 20, check=False)
    receipt_path = png.with_suffix(".json")
    result: dict[str, Any] = {
        "source": str(source),
        "png": str(png),
        "command": command,
        "returncode": process.returncode,
    }
    if not receipt_path.is_file():
        result.update({"valid": False, "error": (process.stderr or process.stdout)[-2000:] or "renderer produced no receipt"})
        return result
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result["receipt"] = receipt
    source_digest = file_sha256(source)
    png_digest = file_sha256(png) if png.is_file() else None
    valid = bool(
        process.returncode == 0
        and receipt.get("valid") is True
        and png.is_file()
        and receipt.get("source_sha256") == source_digest
        and receipt.get("png_sha256") == png_digest
    )
    result.update({
        "valid": valid,
        "source_sha256": source_digest,
        "png_sha256": png_digest,
        "receipt_sha256": file_sha256(receipt_path),
    })
    if not valid:
        result["error"] = "renderer receipt or output hash did not verify"
    return result


def _prepare_render_dirs(root: Path, candidates: Iterable[str], run_as_user: str) -> None:
    """Make isolated output directories writable by the unprivileged renderer."""
    render_root = root / "renders"
    render_root.mkdir(parents=True, exist_ok=True)
    os.chmod(render_root, 0o777)
    try:
        user = pwd.getpwnam(run_as_user)
    except KeyError as exc:
        raise RuntimeError(f"renderer user does not exist: {run_as_user}") from exc
    directories = [render_root]
    for candidate_id in candidates:
        directory = render_root / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        directories.append(directory)
    for directory in directories:
        os.chmod(directory, 0o777)
        try:
            os.chown(directory, user.pw_uid, user.pw_gid)
        except PermissionError as exc:
            raise RuntimeError("render output directory must be chowned to the unprivileged renderer user") from exc
        for child in directory.iterdir():
            if child.is_file() and not child.is_symlink():
                os.chmod(child, 0o666)
                try:
                    os.chown(child, user.pw_uid, user.pw_gid)
                except PermissionError as exc:
                    raise RuntimeError("existing render receipt is not writable by the renderer user") from exc


def renderer_identity(renderer: str | Path) -> dict[str, Any]:
    """Return the source and vendored asset identity of the API renderer."""
    path = Path(renderer).resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"renderer must be a regular file: {path}")
    assets_dir = path.parent / "assets"
    assets: dict[str, str] = {}
    if assets_dir.is_dir():
        for asset in sorted(item for item in assets_dir.rglob("*") if item.is_file() and not item.is_symlink()):
            assets[asset.relative_to(assets_dir).as_posix()] = file_sha256(asset)
    return {
        "renderer_path": path.name,
        "renderer_sha256": file_sha256(path),
        "assets_sha256": sha256_bytes(canonical_json(assets)),
        "assets": assets,
    }


def render(
    output: str | Path,
    renderer: str | Path,
    renderer_python: str | Path,
    *,
    workers: int = 1,
    run_as_user: str = "painter",
    renderer_sha256: str | None = None,
    assets_sha256: str | None = None,
) -> dict[str, Any]:
    """Render all 32 source candidates through the pinned Linux renderer."""
    if sys.platform != "linux":
        raise RuntimeError("Rendering is Linux-node-only; build and metadata preparation are platform-independent.")
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(output).resolve()
    manifest = _read_manifest(root)
    records = _verify_sources(root, manifest)
    _verify_prompt(root, manifest)
    renderer_path = Path(renderer).resolve()
    renderer_python_path = Path(renderer_python).resolve()
    if not renderer_path.is_file() or not renderer_python_path.is_file():
        raise FileNotFoundError("pinned renderer or renderer Python is missing")
    expected_renderer_sha = renderer_sha256 or os.environ.get("PAINTER_RENDERER_SHA256")
    if not isinstance(expected_renderer_sha, str) or not _HASH.fullmatch(expected_renderer_sha):
        raise ValueError("render requires an explicit pinned renderer SHA-256 (or PAINTER_RENDERER_SHA256)")
    identity = renderer_identity(renderer_path)
    if identity["renderer_sha256"] != expected_renderer_sha:
        raise ValueError("pinned renderer source hash mismatch")
    expected_assets_sha = assets_sha256 or os.environ.get("PAINTER_RENDERER_ASSETS_SHA256")
    if expected_assets_sha is not None:
        if not _HASH.fullmatch(expected_assets_sha):
            raise ValueError("renderer assets SHA-256 must be a lowercase 64-character hash")
        if identity["assets_sha256"] != expected_assets_sha:
            raise ValueError("pinned renderer asset hash mismatch")
    _prepare_render_dirs(root, (candidate.id for candidate in CANDIDATES), run_as_user)

    jobs: dict[Any, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in records:
            source = root / record["path"]
            png = root / "renders" / record["candidate_id"] / f"{record['version']}.png"
            future = pool.submit(_renderer_result, source, png, renderer_path, renderer_python_path, run_as_user)
            jobs[future] = record
        for future in as_completed(jobs):
            record = jobs[future]
            try:
                result = future.result()
            except Exception as exc:  # Preserve a censored/failed receipt for auditability.
                result = {"valid": False, "source": record["path"], "error": f"{type(exc).__name__}: {exc}"}
            result.update({"candidate_id": record["candidate_id"], "version": record["version"]})
            results.append(result)
    results.sort(key=lambda item: (item.get("candidate_id", ""), item.get("version", "")))
    valid_by_candidate: dict[str, dict[str, bool]] = {}
    for result in results:
        valid_by_candidate.setdefault(result["candidate_id"], {})[result["version"]] = bool(result.get("valid"))
    render_manifest = {
        "schema": "painter.structured-flow-render.v1",
        "run_id": RUN_ID,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "renderer": identity,
        "program_count": len(records),
        "valid_program_count": sum(bool(r.get("valid")) for r in results),
        "complete_hash_verified": len(results) == 32 and all(bool(r.get("valid")) for r in results),
        "results": results,
        "quality_status": "visual_review_pending",
        "quality_claim": False,
        "notes": "Render validity is an execution/hash check; it is not a visual-improvement judgment.",
    }
    _write_json(root / "render-manifest.json", render_manifest)
    (root / "render-manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in results), encoding="utf-8")
    return render_manifest


def _image_part(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}


def target_block(program: str) -> str:
    """Return the only assistant target form admitted by this run."""
    validate_program(program)
    return "```javascript\n" + program.rstrip() + "\n```"


def _strict_target(target: str, *, terminal: bool = False) -> bool:
    if terminal and target == "FINISHED":
        return True
    return bool(re.fullmatch(r"```javascript\n[\s\S]+\n```", target)) and target.count("```") == 2


def _render_index(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = root / "render-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"render manifest is missing: {path}; run render on Linux first")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "painter.structured-flow-render.v1":
        raise ValueError("invalid structured-flow render manifest")
    renderer = value.get("renderer")
    if not isinstance(renderer, Mapping) or not _HASH.fullmatch(str(renderer.get("renderer_sha256", ""))) or not _HASH.fullmatch(str(renderer.get("assets_sha256", ""))):
        raise ValueError("render manifest is missing the pinned renderer and asset SHA-256 identity")
    if not value.get("complete_hash_verified"):
        raise ValueError("SFT export requires all thirty-two source renders to pass hash verification")
    return {(row["candidate_id"], row["version"]): row for row in value.get("results", [])}


def _candidate_row(manifest: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for row in manifest["rows"]:
        if row["id"] == candidate_id:
            return row
    raise KeyError(candidate_id)


def _user_text(row: Mapping[str, Any], version: str, previous_program: str | None) -> str:
    prior = "none (this is the first complete painting)" if previous_program is None else previous_program
    if version == "block_in":
        canvas = "none yet; the renderer will create it from this program"
    elif version == "improved_final":
        canvas = "the actual rendered block-in canvas attached below"
    else:
        canvas = "the actual rendered final canvas attached below"
    # The native inspect-choice contract is shared by revision and terminal
    # rows.  In particular, a finish row must be allowed to answer FINISHED;
    # the old code-only instruction would make that target contradictory.
    instruction = TEXT_INSPECT_CHOICE_INITIAL if version == "block_in" else TEXT_INSPECT_CHOICE
    return (
        "PAINTING BRIEF\n" + str(row["brief"]) + "\n\n"
        "CURRENT CANVAS\n" + canvas + "\n\n"
        "PREVIOUS PROGRAM\n" + prior + "\n\n"
        + instruction
    )


def _teacher_user_text(
    row: Mapping[str, Any],
    version: str,
    previous_program: str | None,
    prior_canvas: Path | None,
    root: Path,
) -> str:
    """Serialize the multimodal user state for the strict teacher handoff.

    The rich SFT row carries the actual PNG as an image part.  This compact
    string form carries its immutable path and byte hash as well, which lets
    the Linux code-supervision consumer attach the exact image without putting
    a prior assistant target in the conversation.
    """
    base = _user_text(row, version, previous_program)
    if prior_canvas is None:
        canvas = "CURRENT CANVAS FILE: none (first paint)"
    else:
        canvas = (
            "CURRENT CANVAS FILE: " + prior_canvas.relative_to(root).as_posix()
            + "\nCURRENT CANVAS SHA256: " + file_sha256(prior_canvas)
        )
    return canvas + "\n\n" + base


_APPROVAL_STATUSES = {"candidate_approval_pending", "reviewed", "approved", "flash_decision"}
_REVIEW_DECISIONS = {"approved", "flash_decision"}


def _read_approval(
    root: Path,
    approval_manifest: str | Path | Mapping[str, Any] | None,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    """Read an approval manifest without turning a pending file into evidence."""
    if approval_manifest is None:
        if not required:
            return None
        path = root / "approval-candidates.json"
        if not path.is_file():
            raise ValueError("finish targets require actual visual review in approval-candidates.json")
        value = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(approval_manifest, Mapping):
        value = dict(approval_manifest)
    else:
        value = json.loads(Path(approval_manifest).read_text(encoding="utf-8"))
    if value.get("status") not in _APPROVAL_STATUSES:
        raise ValueError("approval manifest has an unknown review status")
    if not isinstance(value.get("candidates"), list):
        raise ValueError("approval manifest has no candidates")
    return value


def _reviewed_receipt(
    candidate: Candidate,
    approval: Mapping[str, Any] | None,
    initial_render: Mapping[str, Any],
    final_render: Mapping[str, Any],
    initial_png: Path,
    final_png: Path,
) -> dict[str, Any] | None:
    """Return a hash-bound review receipt only for a genuine approved item."""
    if approval is None:
        return None
    by_id = {
        str(item.get("id")): item
        for item in approval.get("candidates", [])
        if isinstance(item, Mapping)
    }
    item = by_id.get(candidate.id)
    if item is None:
        return None
    if item.get("decision") not in _REVIEW_DECISIONS or item.get("improved_reviewed") is not True:
        return None
    if item.get("render_valid") is not True:
        raise ValueError(f"reviewed candidate {candidate.id} does not have valid renders")
    actual_initial = file_sha256(initial_png)
    actual_final = file_sha256(final_png)
    if item.get("initial_png_sha256") not in {None, actual_initial}:
        raise ValueError(f"reviewed candidate {candidate.id} initial canvas changed after review")
    if item.get("final_png_sha256") != actual_final:
        raise ValueError(f"reviewed candidate {candidate.id} final canvas changed after review")
    return {
        "verified": True,
        "valid": True,
        "approved": True,
        "source_group": candidate.id,
        "review_status": approval.get("status"),
        "decision": item.get("decision"),
        "improved_reviewed": True,
        "initial_png_sha256": actual_initial,
        "final_png_sha256": actual_final,
        "png_sha256": actual_final,
        "source_sha256": final_render.get("source_sha256"),
        "initial_source_sha256": initial_render.get("source_sha256"),
    }


def _approved_finish_candidates(root: Path, approval_manifest: str | Path | Mapping[str, Any] | None) -> list[Candidate]:
    """Return the first eight easy/connected candidates after explicit review."""
    approval = _read_approval(root, approval_manifest, required=True)
    assert approval is not None
    if approval.get("status") not in {"reviewed", "approved", "flash_decision"}:
        raise ValueError("finish targets require actual visual review or an explicit Flash decision")
    by_id = {str(item.get("id")): item for item in approval.get("candidates", []) if isinstance(item, Mapping)}
    selected = list(CANDIDATES[:8])
    for candidate in selected:
        item = by_id.get(candidate.id)
        if item is None or not item.get("render_valid") or item.get("decision") not in {"approved", "flash_decision"} or item.get("improved_reviewed") is not True:
            raise ValueError(f"finish target {candidate.id} lacks an explicit reviewed improvement decision")
        final_png = root / "renders" / candidate.id / "improved_final.png"
        if file_sha256(final_png) != item.get("final_png_sha256"):
            raise ValueError(f"finish target {candidate.id} final canvas changed after review")
    return selected


def export_sft(
    output: str | Path,
    *,
    include_finish: bool = False,
    approval_manifest: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export 32 code targets, optionally adding 8 reviewed terminal targets."""
    root = Path(output).resolve()
    manifest = _read_manifest(root)
    _verify_sources(root, manifest)
    _verify_prompt(root, manifest)
    render_index = _render_index(root)
    approval = _read_approval(root, approval_manifest, required=include_finish)
    finish_candidates = _approved_finish_candidates(root, approval) if include_finish else []
    reviewed_code_rows = 0
    rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        row = _candidate_row(manifest, candidate.id)
        initial_source = root / row["programs"][0]
        final_source = root / row["programs"][1]
        initial_program = initial_source.read_text(encoding="utf-8")
        final_program = final_source.read_text(encoding="utf-8")
        initial_render = render_index[(candidate.id, "block_in")]
        final_render = render_index[(candidate.id, "improved_final")]
        initial_png = root / "renders" / candidate.id / "block_in.png"
        final_png = root / "renders" / candidate.id / "improved_final.png"
        if not initial_render.get("valid") or not final_render.get("valid"):
            raise ValueError(f"candidate {candidate.id} lacks two hash-verified renders")
        if initial_render.get("source_sha256") != file_sha256(initial_source) or final_render.get("source_sha256") != file_sha256(final_source):
            raise ValueError(f"candidate {candidate.id} source hash does not match render receipt")
        if file_sha256(initial_png) != initial_render.get("png_sha256") or file_sha256(final_png) != final_render.get("png_sha256"):
            raise ValueError(f"candidate {candidate.id} render PNG changed after receipt verification")
        target_receipt = _reviewed_receipt(candidate, approval, initial_render, final_render, initial_png, final_png)
        for version, program, source, png, prior_program, prior_canvas, render_record in (
            ("block_in", initial_program, initial_source, initial_png, None, None, initial_render),
            ("improved_final", final_program, final_source, final_png, initial_program, initial_png, final_render),
        ):
            user_content: list[dict[str, Any]] = [{"type": "text", "text": _user_text(row, version, prior_program)}]
            if prior_canvas is not None:
                user_content.append(_image_part(prior_canvas))
            assistant_target = target_block(program)
            if not _strict_target(assistant_target):
                raise ValueError(f"target for {candidate.id}/{version} is not exactly one JavaScript fence")
            teacher_rows.append({
                "case_id": f"{candidate.id}-{version}",
                "source_group": candidate.id,
                "system": flow_system(),
                "user": _teacher_user_text(row, version, prior_program, prior_canvas, root),
                "next_assistant_token": assistant_target,
            })
            reviewed = target_receipt is not None
            if reviewed:
                reviewed_code_rows += 1
            rows.append({
                "schema": SFT_SCHEMA,
                "id": f"{candidate.id}-{version}",
                "source_group": candidate.id,
                "category": candidate.category,
                "level": candidate.level,
                "split": candidate.split,
                "flow_stage": version,
                "source_program": source.relative_to(root).as_posix(),
                "source_sha256": file_sha256(source),
                "prior_canvas": None if prior_canvas is None else {
                    "path": prior_canvas.relative_to(root).as_posix(),
                    "sha256": file_sha256(prior_canvas),
                },
                "previous_program_in_user_text": prior_program is not None,
                "render": {
                    "png": png.relative_to(root).as_posix(),
                    "png_sha256": render_record.get("png_sha256"),
                    "source_sha256": render_record.get("source_sha256"),
                },
                "target_receipt": None if target_receipt is None else dict(
                    target_receipt,
                    stage=version,
                    png_sha256=file_sha256(png),
                    source_sha256=render_record.get("source_sha256"),
                ),
                "quality_status": "reviewed_improvement" if reviewed else "visual_review_pending",
                "admission": "reviewed_improvement" if reviewed else "render_valid_pending_visual_review",
                "improved_reviewed": reviewed,
                "quality_claim": False,
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": flow_system()}]},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": assistant_target}]},
                ],
            })
    for candidate in finish_candidates:
        row = _candidate_row(manifest, candidate.id)
        final_source = root / row["programs"][1]
        final_program = final_source.read_text(encoding="utf-8")
        final_png = root / "renders" / candidate.id / "improved_final.png"
        final_render = render_index[(candidate.id, "improved_final")]
        if not final_render.get("valid") or final_render.get("source_sha256") != file_sha256(final_source) or final_render.get("png_sha256") != file_sha256(final_png):
            raise ValueError(f"finish target {candidate.id} lacks a final hash-verified canvas")
        initial_render = render_index[(candidate.id, "block_in")]
        initial_png = root / "renders" / candidate.id / "block_in.png"
        target_receipt = _reviewed_receipt(candidate, approval, initial_render, final_render, initial_png, final_png)
        if target_receipt is None:
            raise ValueError(f"finish target {candidate.id} lacks an explicit reviewed improvement decision")
        user_content = [{"type": "text", "text": _user_text(row, "finish", final_program)}, _image_part(final_png)]
        assistant_target = "FINISHED"
        if not _strict_target(assistant_target, terminal=True):
            raise AssertionError("terminal target must be exactly FINISHED")
        teacher_rows.append({
            "case_id": f"{candidate.id}-finish",
            "source_group": candidate.id,
            "system": flow_system(),
            "user": _teacher_user_text(row, "finish", final_program, final_png, root),
            "next_assistant_token": assistant_target,
        })
        rows.append({
            "schema": SFT_SCHEMA,
            "id": f"{candidate.id}-finish",
            "source_group": candidate.id,
            "category": candidate.category,
            "level": candidate.level,
            "split": candidate.split,
            "flow_stage": "finish",
            "source_program": final_source.relative_to(root).as_posix(),
            "source_sha256": file_sha256(final_source),
            "prior_canvas": {"path": final_png.relative_to(root).as_posix(), "sha256": file_sha256(final_png)},
            "previous_program_in_user_text": True,
            "render": {"png": final_png.relative_to(root).as_posix(), "png_sha256": final_render.get("png_sha256"), "source_sha256": final_render.get("source_sha256")},
            "target_receipt": dict(target_receipt, stage="finish", png_sha256=file_sha256(final_png)),
            "quality_status": "reviewed_visual_final",
            "admission": "reviewed_visual_final",
            "improved_reviewed": True,
            "quality_claim": False,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": flow_system()}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_target}]},
            ],
        })
    expected_rows = 32 + len(finish_candidates)
    if len(rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, got {len(rows)}")
    if len(teacher_rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} strict teacher rows, got {len(teacher_rows)}")
    _write_json(root / "sft-manifest.json", {
        "schema": SFT_SCHEMA,
        "run_id": RUN_ID,
        "rows": len(rows),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "native_system_sha256": sha256_text(NATIVE_SYSTEM),
        "quality_status": "visual_review_pending",
        "quality_claim": False,
        "rows_are_assistant_only_targets": True,
        "teacher_jsonl": "teacher.jsonl",
        "teacher_system_sha256": sha256_text(flow_system()),
        "finish_rows": len(finish_candidates),
        "reviewed_code_rows": reviewed_code_rows,
        "approval_manifest_sha256": None if approval is None else (
            file_sha256(Path(approval_manifest))
            if isinstance(approval_manifest, (str, Path))
            else sha256_bytes(canonical_json(approval))
        ),
    })
    (root / "sft.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (root / "teacher.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in teacher_rows), encoding="utf-8")
    for split in ("train", "dev"):
        selected = [row for row in rows if row["split"] == split]
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    return {
        "rows": len(rows),
        "teacher_rows": len(teacher_rows),
        "finish_rows": len(finish_candidates),
        "reviewed_code_rows": reviewed_code_rows,
        "train": sum(r["split"] == "train" for r in rows),
        "dev": sum(r["split"] == "dev" for r in rows),
        "quality_status": "visual_review_pending",
        "quality_claim": False,
    }


def prepare_approval_manifest(output: str | Path) -> dict[str, Any]:
    """Prepare review candidates without silently declaring visual improvement."""
    root = Path(output).resolve()
    manifest = _read_manifest(root)
    render_index = _render_index(root)
    candidates = []
    for row in manifest["rows"]:
        initial = render_index.get((row["id"], "block_in"), {})
        final = render_index.get((row["id"], "improved_final"), {})
        candidates.append({
            "id": row["id"],
            "source_group": row["source_group"],
            "initial_png": f"renders/{row['id']}/block_in.png",
            "final_png": f"renders/{row['id']}/improved_final.png",
            "initial_png_sha256": initial.get("png_sha256"),
            "final_png_sha256": final.get("png_sha256"),
            "render_valid": bool(initial.get("valid") and final.get("valid")),
            "decision": "pending",
            "improved_reviewed": False,
        })
    approval = {
        "schema": "painter.structured-flow-approval-candidates.v1",
        "run_id": RUN_ID,
        "status": "candidate_approval_pending",
        "review_required": "actual visual improvement review or explicit Flash decision",
        "quality_claim": False,
        "candidates": candidates,
    }
    _write_json(root / "approval-candidates.json", approval)
    return approval


def admit_reviewed_corrections(approval: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Admit only explicitly reviewed correction candidates.

    This helper is intentionally strict.  A prepared approval manifest is not
    evidence of improvement; each candidate needs ``decision`` ``approved``
    or ``flash_decision`` and ``improved_reviewed`` true.
    """
    if approval.get("status") not in {"reviewed", "approved", "flash_decision"}:
        raise ValueError("corrections require a reviewed approval manifest or explicit Flash decision")
    candidates = approval.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("approval manifest has no candidates")
    admitted: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("decision") not in {"approved", "flash_decision"} or candidate.get("improved_reviewed") is not True:
            raise ValueError(f"correction candidate {candidate.get('id')} lacks an explicit improvement decision")
        admitted.append(dict(candidate, admission="reviewed_improvement"))
    return admitted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--output", required=True)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--output", "--data", dest="output", required=True)
    render_parser.add_argument("--renderer", required=True)
    render_parser.add_argument("--python", dest="renderer_python", required=True)
    render_parser.add_argument("--renderer-sha256", required=False, help="expected SHA-256 of the pinned renderer.py")
    render_parser.add_argument("--assets-sha256", required=False, help="optional SHA-256 of the pinned renderer asset map")
    render_parser.add_argument("--workers", type=int, default=1)
    render_parser.add_argument("--run-as-user", default="painter")
    export_parser = sub.add_parser("export-sft")
    export_parser.add_argument("--output", "--data", dest="output", required=True)
    export_parser.add_argument("--include-finish", action="store_true", help="add eight reviewed terminal FINISHED rows")
    export_parser.add_argument("--approval", dest="approval_manifest", help="reviewed approval manifest for terminal rows")
    approval_parser = sub.add_parser("prepare-approval")
    approval_parser.add_argument("--output", "--data", dest="output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Keep the convenient historical ``python .../demonstrations.py
    # --output DIR`` shape as a build alias while making all other operations
    # explicit subcommands.
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if argv and argv[0].startswith("--"):
        argv.insert(0, "build")
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        value = build(args.output)
        print(json.dumps({"run_id": RUN_ID, "candidates": value["count"], "programs": value["program_count"], "sft_rows": value["row_count_after_pairing"]}, sort_keys=True))
    elif args.command == "render":
        value = render(
            args.output,
            args.renderer,
            args.renderer_python,
            workers=args.workers,
            run_as_user=args.run_as_user,
            renderer_sha256=args.renderer_sha256,
            assets_sha256=args.assets_sha256,
        )
        print(json.dumps({"run_id": RUN_ID, "programs": value["program_count"], "valid": value["valid_program_count"], "complete_hash_verified": value["complete_hash_verified"]}, sort_keys=True))
    elif args.command == "export-sft":
        print(json.dumps(export_sft(args.output, include_finish=args.include_finish, approval_manifest=args.approval_manifest), sort_keys=True))
    elif args.command == "prepare-approval":
        value = prepare_approval_manifest(args.output)
        print(json.dumps({"run_id": RUN_ID, "candidates": len(value["candidates"]), "status": value["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
