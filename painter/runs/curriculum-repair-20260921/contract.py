"""Shared painting interface. Changes require a new version and matched baselines.

This describes geometry and interaction, not visual quality or the training recipe.
Legacy run directories are deliberately immutable.
"""
import hashlib
import json
import re

VERSION = "canvas-pixels-preview-v2"
CANVAS_SIZE = 600
P5_VERSION = "2.3.2"
BRUSH_VERSION = "2.2.1"

SYSTEM='''Reproduce the supplied reference as an attractive painting using p5.brush JavaScript.
Use p5.js 2.3.2 and p5.brush 2.2.1 with a 600x600 WEBGL canvas. Coordinates initially have origin at the center; translate(-300,-300) once in draw() for a top-left origin.
All geometry uses canvas pixel coordinates: x increases rightward and y downward, with (0,0) at the top left and (600,600) at the bottom right. Do not add per-object affine frames or another coordinate transform. You may use variables, functions, loops, paths and layered brushwork.
Useful brush API: brush.noStroke(); brush.noWash(); brush.fill("#688a9b",180); brush.fillBleed(0.03,"out"); brush.fillTexture(0.2,0.1); brush.polygon([[x1,y1],[x2,y2],[x3,y3]]).
For lines: brush.noFill(); brush.noWash(); brush.set("cpencil","#554d44",0.8); brush.line(x1,y1,x2,y2). Reset inactive paint modes when switching. Prefer brush.fill for polygons; wash-to-pencil transitions have lost a pending face in this runtime.
Inspect the reference and current canvas. Preserve correct structure while improving the painting.
Do not embed images, copy pixels, write labels in place of objects, or access external resources.
There is no target stroke count or speed reward in this quality phase. Finish when the painting is good.
Return a brief visual plan followed by a complete sketch inside a javascript code block.
Use setup() and draw(); finish draw() with noLoop(). You will receive the actual render to revise.
Painting and finishing are separate actions. After submitting code, inspect its actual render on the next turn. Never append FINISHED to a code response. Only finish after observing a CURRENT CANVAS that matches the reference.
If the CURRENT CANVAS already faithfully matches the reference, keep it unchanged: give a brief reason and FINISHED without a code block. This is allowed only when a current canvas exists.
Do not repeat an unchanged sketch as a revision. Preserve correct parts and fix the specific remaining mismatch.'''

NEXT_VERSION='Inspect the images and paint the next complete version.'
NO_CHANGE='Your previous revision made no change to the current canvas. Inspect the reference and current canvas. If the painting is already complete, keep it and respond FINISHED without code. Otherwise identify a specific visual mismatch and change the sketch to fix it; do not repeat the same program.'
FINISH_WITHOUT_CANVAS='Do not respond FINISHED yet: this task has no observed canvas. Return a complete JavaScript sketch in a javascript code block so it can be rendered, then inspect the actual canvas before deciding whether to finish.'


def identity():
    return {"version": VERSION, "sha256": hashlib.sha256(
        json.dumps({"version": VERSION, "canvas": CANVAS_SIZE,
                    "p5": P5_VERSION, "brush": BRUSH_VERSION,
                    "system": SYSTEM, "next": NEXT_VERSION, "no_change": NO_CHANGE,
                    "finish_without_canvas": FINISH_WITHOUT_CANVAS},
                   sort_keys=True).encode()).hexdigest()}


def paint_target(plan, program):
    validate_teacher_program(program)
    return plan + "\n```javascript\n" + program.rstrip() + "\n```"


def finish_target():
    return "The observed canvas matches the reference. Keep it unchanged.\nFINISHED"


def validate_teacher_program(program):
    """Catch known incompatible teacher formats; NOT a JS sandbox or full parser.

    This is only a data-authoring check. Student mistakes are rendered and exposed
    as feedback, never silently rewritten or rejected as low-quality paintings.
    """
    if re.search(r"\b(?:let|const|var)\s+frame\s*=|\bfunction\s+mapped\s*\(", program):
        raise ValueError("Legacy affine teacher program: compile to canvas pixels before SFT")
    translations = re.findall(r"\btranslate\s*\(([^)]*)\)", program)
    if len(translations) != 1 or re.sub(r"\s", "", translations[0]) != "-300,-300":
        raise ValueError("Teacher must translate(-300,-300) exactly once")
    if re.search(r"(?<![\w.])(?:scale|rotate|applyMatrix|resetMatrix)\s*\(", program):
        raise ValueError("Teacher geometry must use canvas pixel coordinates")
