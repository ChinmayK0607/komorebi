#!/usr/bin/env python3
"""Hand-authored p5.brush studies of ten screened harder references.

Each scene is composed semantically by hand, with no pixel sampling, embedded
source image, or automatic vector tracing. Run on a Linux renderer and inspect
the actual canvases before admitting any version as teacher data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
OUT = ROOT / "painter/collected/quality-curriculum-20260924/manual"


class Painting:
    def __init__(self, reference: str, background: str):
        self.reference = reference
        self.background = background
        self.shapes: list[dict] = []

    def poly(self, points, color, alpha=220, texture=0.2):
        self.shapes.append(dict(t="poly", p=points, c=color, a=alpha, x=texture))

    def ellipse(self, cx, cy, rx, ry, color, alpha=220, texture=0.2, tilt=0):
        self.shapes.append(dict(t="ellipse", cx=cx, cy=cy, rx=rx, ry=ry,
                                c=color, a=alpha, x=texture, tilt=tilt))

    def line(self, points, color, width=1):
        self.shapes.append(dict(t="line", p=points, c=color, w=width))


def cake() -> Painting:
    p = Painting("coco128-000000000092", "#b8a896")
    p.poly([[0,0],[600,0],[600,600],[0,600]], "#ae9b86", 230, .27)
    p.poly([[0,305],[600,235],[600,600],[0,600]], "#cdbba7", 225, .24)
    p.ellipse(330,400,285,145,"#ddd8cf",235,.11)
    p.ellipse(332,390,242,111,"#f7f3eb",240,.06)
    p.ellipse(340,407,208,80,"#d1c8ba",110,.06)
    # Top, front and dark left plane of the chocolate layer cake.
    p.poly([[134,205],[393,173],[470,237],[216,264]],"#422319",245,.13)
    p.poly([[216,264],[470,237],[463,425],[224,451]],"#67402d",245,.22)
    p.poly([[134,205],[216,264],[224,451],[142,395]],"#3b211c",245,.16)
    p.poly([[229,284],[458,260],[458,300],[228,323]],"#ab8065",210,.17)
    p.poly([[228,331],[458,306],[458,334],[229,360]],"#34211d",235,.12)
    p.poly([[230,366],[458,341],[459,372],[231,398]],"#b18b71",210,.17)
    p.poly([[232,409],[458,381],[460,419],[232,442]],"#3e2520",238,.12)
    p.poly([[148,217],[210,271],[214,434],[147,389]],"#4e2b23",130,.18)
    p.poly([[155,212],[394,183],[461,231],[215,255]],"#5b2c1b",220,.25)
    p.poly([[171,206],[363,185],[434,223],[217,245]],"#743d27",130,.3)
    # Coconut flecks, deliberately sparse rather than repeated texture noise.
    for x,y,w in [(228,207,11),(252,202,9),(279,193,13),(305,211,10),(341,200,12),
                  (379,218,10),(314,222,8),(206,222,10),(267,228,8)]:
        p.ellipse(x,y,w,2.8,"#f2e4c9",210,.08,tilt=-.25)
    # Fork on the right, a clean scale cue.
    p.poly([[466,291],[478,287],[526,514],[516,517]],"#b1a07c",210,.07)
    p.line([[474,300],[526,513]],"#fff4d6",1.2)
    for x in [460,470,480,490]:
        p.line([[x,286],[x+1,252]],"#665b4b",1.4)
    p.line([[450,287],[490,287]],"#665b4b",1.4)
    return p


def bus() -> Painting:
    p = Painting("coco128-000000000471", "#dcebf0")
    p.poly([[0,0],[600,0],[600,301],[0,308]],"#c6e3ee",225,.18)
    p.poly([[0,302],[600,297],[600,600],[0,600]],"#c9c9bb",235,.17)
    p.poly([[0,373],[600,368],[600,600],[0,600]],"#b7b7ad",125,.21)
    p.ellipse(322,440,258,37,"#4e5150",115,.11)
    # Three-quarter body: long right side and angled front cap.
    p.poly([[184,196],[490,210],[523,245],[526,384],[180,405]],"#d4a33e",245,.2)
    p.poly([[115,237],[184,196],[180,405],[111,380]],"#c68e32",245,.17)
    p.poly([[142,210],[194,183],[476,198],[500,214],[184,203]],"#f4e5bc",225,.17)
    p.poly([[128,234],[179,208],[174,320],[120,339]],"#dde8e7",225,.07)
    p.poly([[188,216],[488,228],[497,288],[185,278]],"#3e5662",230,.12)
    # Window dividers and reflected sky within the glass band.
    for x in [218,262,307,351,396,440]:
        p.poly([[x,219+(x-188)*.04],[x+5,220+(x-188)*.04],
                [x+5,281+(x-188)*.02],[x,281+(x-188)*.02]],"#d9b56e",230,.07)
    for x in [224,313,401]:
        p.poly([[x,226],[x+31,229],[x+28,248],[x+3,245]],"#91aab1",100,.04)
    p.poly([[182,291],[521,296],[525,351],[181,365]],"#dcaa42",235,.18)
    p.poly([[181,363],[526,348],[523,385],[181,406]],"#bd822f",210,.22)
    p.line([[113,348],[180,367],[523,352]],"#695338",1.2)
    p.line([[183,288],[516,301]],"#7b643f",1.3)
    p.poly([[116,345],[176,364],[175,397],[116,379]],"#a8722c",205,.16)
    # Recessed wheel wells, strong round wheels, hubs.
    for x,y,r in [(220,395,42),(462,382,35)]:
        p.ellipse(x,y,r+7,r+7,"#745c38",235,.14)
        p.ellipse(x,y,r,r,"#292b2b",245,.09)
        p.ellipse(x,y,r*.43,r*.43,"#c5a164",230,.06)
        p.ellipse(x,y,r*.16,r*.16,"#77634d",220,.05)
    p.ellipse(130,354,11,15,"#f0e9cf",230,.07)
    p.ellipse(174,348,7,10,"#e6d5a2",210,.07)
    p.line([[118,384],[180,405],[523,384]],"#484b45",1.1)
    p.line([[98,406],[154,412]],"#f6f5e8",1.2)
    p.line([[365,485],[537,483]],"#eeece3",2.3)
    return p


def dog_frisbee() -> Painting:
    p = Painting("coco128-000000000394", "#627345")
    p.poly([[0,0],[600,0],[600,600],[0,600]],"#536e44",230,.24)
    p.poly([[0,398],[600,360],[600,600],[0,600]],"#749051",215,.24)
    for x,y in [(38,504),(105,559),(214,488),(455,533),(548,468)]:
        p.line([[x,y+20],[x+8,y-5],[x+10,y+14]],"#a1ad6d",.8)
    p.ellipse(232,337,230,103,"#f0eee2",235,.18)
    p.ellipse(252,322,200,69,"#ffffff",145,.2)
    p.poly([[73,320],[28,291],[5,267],[0,319],[83,364]],"#e4e4d9",220,.18)
    p.poly([[145,397],[177,398],[173,557],[135,553]],"#e9e5db",235,.16)
    p.poly([[282,397],[320,393],[322,552],[286,556]],"#eeeae0",235,.15)
    p.poly([[421,395],[448,387],[458,542],[422,545]],"#eae8dc",240,.18)
    # Head and ears are placed before the foreground disc.
    p.ellipse(467,255,112,96,"#f1efe4",245,.21)
    p.ellipse(417,256,39,66,"#dedbce",225,.21,tilt=.38)
    p.ellipse(538,261,29,64,"#d6d4c9",225,.21,tilt=-.3)
    p.ellipse(480,302,63,42,"#f7f4e9",235,.16)
    p.ellipse(449,254,9,11,"#322e29",250,.04)
    p.ellipse(512,255,10,11,"#322e29",250,.04)
    p.ellipse(483,295,18,13,"#302c2c",250,.05)
    p.line([[481,306],[474,318]],"#55423c",.8)
    # Purple disc must visibly occlude chest but not hide the dog's face.
    p.ellipse(406,391,165,104,"#49337d",245,.11,tilt=-.24)
    p.ellipse(403,379,152,91,"#614390",240,.15,tilt=-.24)
    p.ellipse(408,376,119,69,"#6c4ca0",150,.13,tilt=-.24)
    p.ellipse(408,378,81,46,"#47377b",95,.09,tilt=-.24)
    p.line([[481,315],[504,326],[520,351]],"#514643",.7)
    return p


def teddy() -> Painting:
    p = Painting("coco128-000000000491", "#a39175")
    p.poly([[0,0],[600,0],[600,600],[0,600]],"#91836e",235,.25)
    p.ellipse(90,130,155,185,"#b4a58d",170,.25)
    p.ellipse(520,205,160,225,"#c5b195",160,.24)
    p.poly([[0,415],[600,408],[600,600],[0,600]],"#6e6559",180,.23)
    p.ellipse(300,547,185,36,"#4c4945",85,.11)
    # Bear: readable limbs behind a generous plush torso.
    p.ellipse(178,423,91,151,"#eee9dd",235,.29,tilt=-.18)
    p.ellipse(425,428,88,150,"#e8e2d4",235,.29,tilt=.18)
    p.ellipse(295,385,158,186,"#f3f0e7",245,.29)
    p.ellipse(188,376,73,121,"#f2ece0",235,.3,tilt=.2)
    p.ellipse(412,374,73,121,"#e9e2d7",235,.3,tilt=-.2)
    p.ellipse(287,545,73,65,"#ede7de",240,.22)
    p.ellipse(393,542,67,62,"#ebe4d9",240,.22)
    p.ellipse(287,557,32,29,"#cead9b",175,.18)
    p.ellipse(393,552,30,28,"#c9a898",175,.18)
    # Two ears then face; eye spacing and muzzle stay coherent.
    p.ellipse(205,155,55,60,"#ede8df",235,.24)
    p.ellipse(400,155,55,60,"#eae5db",235,.24)
    p.ellipse(206,161,29,32,"#d9c3b5",185,.15)
    p.ellipse(398,162,29,32,"#d9c5b7",185,.15)
    p.ellipse(301,225,123,120,"#f6f3ed",245,.28)
    p.ellipse(307,262,69,49,"#e8e2d7",232,.23)
    p.ellipse(257,217,10,12,"#302d2c",245,.04)
    p.ellipse(347,217,10,12,"#302d2c",245,.04)
    p.ellipse(303,250,17,11,"#473d39",240,.06)
    p.line([[303,257],[301,274],[289,279]],"#62534a",.9)
    p.line([[301,274],[315,281]],"#62534a",.9)
    # Short red ribbon is a distinct feature of the source toy.
    p.poly([[294,341],[250,329],[267,361],[299,348]],"#ba5149",200,.14)
    p.poly([[300,342],[346,326],[330,362],[298,351]],"#a84743",200,.14)
    p.ellipse(299,346,10,11,"#7a3838",230,.06)
    return p


def coffee() -> Painting:
    p = Painting("coco128-000000000605", "#8d593d")
    p.poly([[0,0],[600,0],[600,600],[0,600]],"#a96b48",235,.32)
    for y in [39,111,195,275,369,459,532]:
        p.line([[0,y+14],[270,y-7],[600,y+19]],"#71452f",1.25)
        p.line([[0,y+22],[310,y+1],[600,y+28]],"#db9a68",.7)
    p.ellipse(350,366,196,145,"#47392f",95,.12,tilt=-.08)
    p.ellipse(348,355,181,136,"#25465b",238,.17,tilt=-.08)
    p.ellipse(352,354,147,100,"#547284",170,.16,tilt=-.08)
    # White cup and coffee top, with a true visible handle.
    p.ellipse(426,324,51,43,"#f7eee1",235,.08)
    p.ellipse(432,324,28,25,"#638093",205,.08)
    p.poly([[222,274],[452,270],[423,383],[248,391]],"#e9e5d9",240,.14)
    p.ellipse(336,275,119,74,"#f6f1e7",245,.08)
    p.ellipse(335,277,102,60,"#cba76a",230,.11)
    p.ellipse(336,274,89,50,"#e4ca89",215,.09)
    # Minimal latte heart, deliberately fluid rather than icon-like.
    p.ellipse(319,269,22,13,"#fff0c3",220,.08,tilt=-.25)
    p.ellipse(349,270,22,13,"#fff0c3",220,.08,tilt=.25)
    p.poly([[303,270],[335,297],[367,270]],"#fff0c3",220,.07)
    p.line([[335,290],[352,300]],"#f6e4b4",1.1)
    # Spoon and pastry accents make it a café still life.
    p.ellipse(229,395,22,12,"#bab8ae",200,.08,tilt=-.3)
    p.poly([[224,395],[233,396],[279,480],[272,483]],"#d4d0c5",220,.07)
    p.ellipse(90,214,93,70,"#f5ebe0",210,.09)
    p.ellipse(91,216,75,52,"#573426",190,.18)
    p.ellipse(88,206,47,28,"#a85c37",195,.23)
    p.poly([[0,280],[169,258],[183,332],[0,364]],"#ece3d4",105,.19)
    return p


def cat_car() -> Painting:
    p = Painting("coco128-000000000650", "#797b70")
    p.poly([[0,0],[600,0],[600,405],[0,415]],"#7b8176",225,.31)
    for x,y,r in [(71,148,60),(186,98,65),(354,140,79),(543,105,72)]:
        p.ellipse(x,y,r,r*.75,"#526651",105,.29)
    p.poly([[0,416],[205,386],[600,401],[600,600],[0,600]],"#253748",245,.14)
    p.poly([[0,487],[600,469],[600,600],[0,600]],"#182937",235,.12)
    p.line([[0,484],[366,474],[600,488]],"#9ea7a0",1.4)
    p.ellipse(326,385,159,32,"#171c21",120,.09)
    # Compact tabby torso, haunches, tail, and four grounded paws.
    p.ellipse(347,273,147,95,"#746f66",238,.23)
    p.ellipse(392,323,83,72,"#77736c",230,.21)
    p.poly([[204,275],[168,270],[142,292],[159,320],[217,320]],"#585952",225,.2)
    p.poly([[282,337],[315,343],[300,389],[274,392]],"#eee9e2",230,.15)
    p.poly([[361,339],[395,339],[398,390],[372,390]],"#eee9e2",230,.15)
    p.poly([[423,323],[455,315],[464,385],[436,389]],"#d7d4ce",225,.16)
    p.ellipse(290,391,33,13,"#f7f1e9",230,.08)
    p.ellipse(386,392,31,13,"#f7f1e9",230,.08)
    p.ellipse(449,387,29,12,"#f6eee5",225,.08)
    # Head, upright triangular ears and two wide-set eyes.
    p.ellipse(251,243,72,67,"#8a8379",240,.22)
    p.poly([[190,215],[203,155],[232,197]],"#7a756b",235,.17)
    p.poly([[274,194],[309,156],[305,227]],"#78766b",235,.17)
    p.poly([[199,205],[207,173],[224,199]],"#c59d94",160,.09)
    p.poly([[279,199],[300,175],[297,211]],"#bd9b91",160,.09)
    p.ellipse(249,273,49,34,"#f3eee6",225,.17)
    p.ellipse(219,241,8,10,"#9d9b62",235,.06)
    p.ellipse(279,240,8,10,"#9d9b62",235,.06)
    p.ellipse(218,240,3,8,"#282d25",245,.03)
    p.ellipse(279,239,3,8,"#282d25",245,.03)
    p.poly([[244,262],[257,262],[250,272]],"#a37571",230,.06)
    p.line([[251,271],[249,279],[239,283]],"#544d49",.65)
    p.line([[249,279],[260,283]],"#544d49",.65)
    for y in [271,279,286]:
        p.line([[229,y],[179,y-12]],"#efe9df",.45)
        p.line([[269,y],[322,y-10]],"#efe9df",.45)
    # A few broad stripes follow the body curvature, not a uniform hatch.
    for x,y in [(273,238),(314,221),(357,217),(399,229),(444,246)]:
        p.poly([[x,y],[x+19,y+8],[x+9,y+53],[x-4,y+47]],"#424a46",85,.17)
    return p


def zebra() -> Painting:
    p = Painting("coco128-000000000034", "#9db568")
    p.poly([[0,0],[600,0],[600,600],[0,600]],"#a6bd74",245,.29)
    for x,y,r in [(42,88,47),(165,70,38),(297,99,42),(510,73,46),(586,153,54)]:
        p.ellipse(x,y,r,r*.57,"#6f9654",110,.22)
    p.poly([[0,435],[600,405],[600,600],[0,600]],"#7fa45b",175,.3)
    p.ellipse(330,480,254,33,"#536d4a",68,.1)
    # Grazing zebra: shoulder and barrel cropped on the left, downward neck.
    p.poly([[0,188],[227,160],[365,200],[382,280],[294,345],[0,350]],"#e8e4d4",245,.18)
    p.poly([[287,229],[348,225],[440,264],[479,319],[421,356],[341,310]],"#e6e1d0",245,.17)
    p.poly([[423,290],[470,277],[529,337],[522,444],[484,464],[454,378]],"#f0eadb",245,.16)
    p.poly([[475,385],[523,371],[544,423],[509,477],[478,467]],"#eee6d5",245,.15)
    p.poly([[503,460],[548,452],[539,496],[510,512],[480,486]],"#746c61",232,.13)
    p.poly([[504,331],[513,286],[531,293],[533,359]],"#efe9da",235,.14)
    p.poly([[464,313],[466,269],[484,263],[494,322]],"#e5dfd1",235,.13)
    # Four distinct grounded legs; rear legs are partially obscured.
    for points in [
        [[70,335],[108,342],[119,512],[89,518]],
        [[176,334],[205,339],[195,505],[166,505]],
        [[302,329],[331,330],[346,507],[318,513]],
        [[358,316],[384,320],[414,502],[388,509]],
    ]:
        p.poly(points,"#e4dece",235,.14)
    for x,y in [(89,515),(168,506),(320,511),(390,507)]:
        p.poly([[x,y-4],[x+32,y-6],[x+30,y+8],[x-2,y+8]],"#4d4d49",230,.07)
    # Mane and tail set the silhouette; body stripes follow angled anatomy.
    p.poly([[231,166],[255,159],[388,249],[376,274]],"#383936",225,.16)
    p.line([[4,305],[-8,361]],"#363631",4)
    for x,yt,yb,lean in [(20,192,340,25),(62,179,344,32),(110,171,346,29),
                         (159,165,343,39),(211,166,337,43),(266,182,322,34)]:
        p.poly([[x,yt],[x+19,yt+3],[x+lean+8,yb-10],[x+lean-6,yb]],"#30302f",220,.11)
    for x,y in [(304,229),(325,239),(344,250),(366,268)]:
        p.poly([[x,y],[x+16,y+7],[x+55,y+55],[x+35,y+46]],"#343431",215,.09)
    for x,y in [(462,298),(478,321),(493,344),(509,370),(503,397)]:
        p.poly([[x,y],[x+17,y+3],[x+29,y+31],[x+11,y+27]],"#343635",220,.08)
    for x,y in [(92,366),(179,360),(319,350),(383,350)]:
        p.poly([[x,y],[x+16,y+3],[x+17,y+49],[x+2,y+40]],"#383936",205,.09)
    p.ellipse(500,359,5,6,"#242b27",235,.04)
    p.line([[540,484],[516,497]],"#363430",1.3)
    return p


def elephants() -> Painting:
    p = Painting("coco128-000000000263", "#70866b")
    p.poly([[0,0],[600,0],[600,450],[0,458]],"#688265",220,.3)
    for x,y,r in [(36,96,97),(203,74,85),(428,75,113),(565,128,81)]:
        p.ellipse(x,y,r,r*.68,"#486a53",108,.28)
    p.poly([[0,443],[600,423],[600,600],[0,600]],"#b4a697",220,.21)
    p.ellipse(350,543,235,45,"#5b5a53",78,.13)
    # Smaller calf behind, reaching across the larger elephant's back.
    p.ellipse(162,263,174,141,"#777474",235,.21)
    p.ellipse(262,241,82,85,"#807c7a",235,.2)
    p.ellipse(246,265,44,66,"#6a696a",190,.18,tilt=.3)
    p.poly([[250,277],[309,265],[362,329],[341,358]],"#716e6d",238,.2)
    p.poly([[48,340],[111,338],[135,470],[92,473]],"#716f70",235,.18)
    p.poly([[149,350],[205,350],[210,463],[170,470]],"#747171",235,.19)
    p.ellipse(272,218,6,7,"#333837",235,.04)
    # Foreground calf: broad barrel, four legs, ears, curved trunk.
    p.ellipse(378,367,191,135,"#98908a",242,.22)
    p.poly([[231,430],[292,432],[305,563],[261,566]],"#8e8885",240,.2)
    p.poly([[323,429],[374,432],[375,562],[329,563]],"#99928e",242,.18)
    p.poly([[431,434],[483,433],[482,557],[443,558]],"#85817f",240,.19)
    p.poly([[492,422],[534,414],[548,540],[514,544]],"#88817e",238,.19)
    p.ellipse(499,326,99,94,"#918984",242,.2)
    p.ellipse(449,341,52,67,"#797674",223,.18,tilt=-.28)
    p.poly([[536,337],[562,349],[582,416],[570,478],[546,483],[550,447],[534,400]],"#8d8783",238,.18)
    p.ellipse(565,477,17,19,"#7b7773",212,.11)
    p.ellipse(528,316,6,8,"#292e2c",238,.04)
    p.poly([[459,327],[489,303],[486,381],[460,386]],"#a8a09b",115,.17)
    p.line([[447,396],[503,397]],"#726f6d",.9)
    p.line([[337,330],[358,309],[394,309]],"#c4b5a9",1.1)
    for x,y in [(262,561),(332,558),(449,552),(520,538)]:
        p.line([[x,y],[x+31,y+2]],"#676665",.8)
    return p


def pontoon_dog() -> Painting:
    p = Painting("coco128-000000000400", "#b8c0bf")
    p.poly([[0,0],[600,0],[600,381],[0,389]],"#c3c8c8",230,.16)
    p.poly([[0,384],[600,372],[600,600],[0,600]],"#3d514f",240,.24)
    for y in [413,451,495,541,578]:
        p.line([[0,y],[140,y+5],[290,y-3],[470,y+8],[600,y+1]],"#9da9a1",.55)
    p.ellipse(288,470,325,37,"#253734",95,.11)
    # Hull and pontoon: large simple grays with selected metal highlights.
    p.poly([[33,365],[583,342],[548,454],[96,486]],"#7b8585",238,.13)
    p.poly([[73,409],[552,383],[522,455],[105,474]],"#5f6a6b",235,.16)
    p.poly([[12,287],[588,278],[585,378],[25,398]],"#e9e8e3",240,.12)
    p.poly([[28,370],[584,354],[583,382],[38,408]],"#a9adab",220,.1)
    p.line([[18,384],[583,365]],"#f8f4e8",1.3)
    # Cabin: two wide panels and guardrail, dog in the right-side opening.
    p.poly([[69,202],[404,198],[451,222],[451,337],[67,351]],"#dfdfda",238,.12)
    p.poly([[72,219],[406,215],[429,235],[427,320],[74,337]],"#bfc5c3",225,.12)
    p.poly([[437,226],[534,225],[545,321],[449,329]],"#d9dcda",238,.11)
    p.poly([[451,235],[523,235],[528,299],[452,303]],"#6a7774",235,.11)
    p.poly([[73,323],[543,310],[548,354],[67,370]],"#f0eee9",238,.1)
    p.line([[48,215],[423,209],[538,219]],"#777e7b",1.25)
    for x in [107,295,388,438,548]:
        p.line([[x,219],[x,353]],"#8b9290",.8)
    p.ellipse(478,274,42,37,"#987661",235,.18)
    p.ellipse(460,245,21,32,"#7d6051",218,.15,tilt=-.32)
    p.ellipse(505,247,18,30,"#7f6255",218,.15,tilt=.3)
    p.ellipse(492,282,27,20,"#b69278",225,.12)
    p.ellipse(493,270,4,5,"#302c29",240,.04)
    p.ellipse(517,286,6,5,"#463831",230,.04)
    p.line([[0,181],[92,179],[420,180]],"#858b89",1.2)
    p.line([[29,180],[29,224]],"#808784",1)
    return p


def giraffes() -> Painting:
    p = Painting("coco128-000000000025", "#9caa78")
    p.poly([[0,0],[600,0],[600,600],[0,600]],"#9bad80",230,.24)
    for x,y,r in [(44,162,109),(183,128,125),(498,119,131),(578,218,102)]:
        p.ellipse(x,y,r,r*.9,"#526745",160,.33)
    p.poly([[0,412],[600,393],[600,600],[0,600]],"#899264",210,.28)
    # Massive tree is a visual anchor behind the animal.
    p.poly([[268,0],[329,0],[362,423],[314,435]],"#6a5743",235,.21)
    p.poly([[313,156],[374,118],[486,90],[498,108],[358,214]],"#6d5d46",225,.19)
    p.poly([[317,243],[232,207],[146,173],[137,192],[301,278]],"#725c43",225,.2)
    p.line([[288,6],[337,418]],"#b8a274",1.1)
    p.poly([[0,487],[600,466],[600,538],[0,550]],"#795e44",225,.18)
    # Tall main giraffe: light body, four legs, long neck and small head.
    p.poly([[346,300],[424,259],[502,281],[532,367],[461,407],[377,386]],"#c89055",240,.2)
    p.poly([[398,284],[415,110],[448,93],[452,294]],"#d5a365",245,.16)
    p.poly([[414,110],[438,70],[465,74],[475,105],[449,127]],"#d5a467",240,.16)
    p.poly([[430,88],[422,43],[433,43],[443,80]],"#c0915e",230,.12)
    p.poly([[455,87],[458,45],[469,50],[467,90]],"#c0925f",230,.12)
    p.ellipse(421,44,7,8,"#5c4d39",230,.06)
    p.ellipse(468,49,7,8,"#5b4a3a",230,.06)
    p.ellipse(447,93,5,6,"#342f2b",240,.04)
    p.ellipse(474,105,8,6,"#72533b",240,.05)
    for pts in [
        [[371,376],[397,376],[392,536],[368,535]],
        [[407,382],[428,383],[438,535],[417,534]],
        [[469,387],[492,379],[508,530],[486,535]],
        [[508,366],[529,360],[550,527],[529,531]],
    ]:
        p.poly(pts,"#c2915c",235,.18)
    for x in [369,417,487,530]:
        p.poly([[x,528],[x+22,527],[x+24,543],[x-2,543]],"#574536",230,.07)
    # Purposeful sparse patches, respecting the neck and barrel outline.
    for x,y,r in [(399,142,8),(427,177,10),(401,217,9),(433,251,10),
                  (378,310,13),(416,319,14),(458,314,13),(493,331,12),
                  (392,352,12),(445,357,14),(480,370,10)]:
        p.ellipse(x,y,r,r*.77,"#9a603a",200,.12,tilt=.18)
    p.line([[521,327],[563,383]],"#5d4b35",1.7)
    # A second distant giraffe partly entering at bottom left, as in source.
    p.poly([[2,497],[67,461],[145,472],[151,503],[0,522]],"#bc925f",200,.17)
    p.poly([[112,475],[139,405],[152,409],[143,485]],"#bb8c54",190,.14)
    p.ellipse(151,409,18,12,"#be925d",205,.12)
    for x,y in [(27,487),(57,479),(84,487),(128,448)]:
        p.ellipse(x,y,8,6,"#805737",165,.09)
    return p


SCENES = {"coco128-000000000092": cake,
          "coco128-000000000025": giraffes,
          "coco128-000000000034": zebra,
          "coco128-000000000263": elephants,
          "coco128-000000000394": dog_frisbee,
          "coco128-000000000400": pontoon_dog,
          "coco128-000000000471": bus,
          "coco128-000000000491": teddy,
          "coco128-000000000605": coffee,
          "coco128-000000000650": cat_car}

TEMPLATE = r'''function setup(){createCanvas(600,600,WEBGL);pixelDensity(1);randomSeed(149);noiseSeed(149);brush.seed(149);background(BG);}
function pigment(pts,color,alpha,texture){brush.noStroke();brush.noWash();brush.fill(color,alpha);brush.fillBleed(.025,"out");brush.fillTexture(texture,texture*.45);brush.polygon(pts);}
function roundMark(s){const pts=[];for(let i=0;i<40;i++){const t=2*Math.PI*i/40,ct=Math.cos(s.tilt||0),st=Math.sin(s.tilt||0),u=s.rx*Math.cos(t),v=s.ry*Math.sin(t);pts.push([s.cx+u*ct-v*st,s.cy+u*st+v*ct]);}pigment(pts,s.c,s.a,s.x);}
function pencil(points,color,width){brush.noFill();brush.noWash();brush.set("cpencil",color,width);for(let i=1;i<points.length;i++)brush.line(points[i-1][0],points[i-1][1],points[i][0],points[i][1]);}
function draw(){translate(-300,-300);const layers=LAYERS;for(const s of layers){if(s.t==="poly")pigment(s.p,s.c,s.a,s.x);else if(s.t==="ellipse")roundMark(s);else if(s.t==="line")pencil(s.p,s.c,s.w);}noLoop();}
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "reference-manifest.json").read_text(encoding="utf-8"))
    refs = {row["id"]: row for row in manifest["references"]}
    output = []
    for ident, factory in SCENES.items():
        if ident not in refs or refs[ident]["tier"] != "manual hard":
            raise ValueError(f"unapproved manual source: {ident}")
        painting = factory()
        if painting.reference != ident:
            raise ValueError(f"factory identity mismatch: {ident}")
        code = "const BG=" + json.dumps(painting.background) + ";\n"
        code += "const LAYERS=" + json.dumps(painting.shapes, separators=(",", ":")) + ";\n"
        code += TEMPLATE
        folder = OUT / ident
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "v1.js"
        path.write_text(code, encoding="utf-8")
        output.append({"reference_id": ident, "reference_sha256": refs[ident]["sha256"],
                       "program_path": str(path.relative_to(ROOT)),
                       "program_sha256": hashlib.sha256(code.encode()).hexdigest(),
                       "shape_count": len(painting.shapes),
                       "status": "authored_unrendered"})
    (RUN / "manual-program-manifest.json").write_text(json.dumps({
        "schema": "painter.manual-teachers.v1", "count": len(output), "programs": output,
        "note": "Authored programs are not teacher data until Linux render, visual inspection, and revision."
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Compiled {len(output)} self-contained unrendered programs")


if __name__ == "__main__":
    main()
