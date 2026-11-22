"""Quest Pro (Meta face tracking v2, 70 weights) -> Babble face model (45 outputs).

BABBLE = output order of faceModel.onnx = ParameterSenderService.FaceExpressionMap in Baballonia.
MAP says which Quest Pro weights are averaged to make the target for each Babble output.
Outputs with no entry (11 tongue shapes) have NO teacher: training keeps them at the stock model's values.
"""
import numpy as np

BABBLE = ["CheekPuffLeft", "CheekPuffRight", "CheekSuckLeft", "CheekSuckRight", "JawOpen", "JawForward", "JawLeft",
          "JawRight", "NoseSneerLeft", "NoseSneerRight", "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight",
          "MouthRollUpper", "MouthRollLower", "MouthShrugUpper", "MouthShrugLower", "MouthClose", "MouthSmileLeft",
          "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight",
          "MouthUpperUpLeft", "MouthUpperUpRight", "MouthLowerDownLeft", "MouthLowerDownRight", "MouthPressLeft",
          "MouthPressRight", "MouthStretchLeft", "MouthStretchRight", "TongueOut", "TongueUp", "TongueDown",
          "TongueLeft", "TongueRight", "TongueRoll", "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat",
          "TongueTwistLeft", "TongueTwistRight"]

MAP = {
    "CheekPuffLeft": ["CheekPuffL"], "CheekPuffRight": ["CheekPuffR"],
    "CheekSuckLeft": ["CheekSuckL"], "CheekSuckRight": ["CheekSuckR"],
    "JawOpen": ["JawDrop"], "JawForward": ["JawThrust"],
    "JawLeft": ["JawSidewaysLeft"], "JawRight": ["JawSidewaysRight"],
    "NoseSneerLeft": ["NoseWrinklerL"], "NoseSneerRight": ["NoseWrinklerR"],
    "MouthFunnel": ["LipFunnelerLb", "LipFunnelerLt", "LipFunnelerRb", "LipFunnelerRt"],
    "MouthPucker": ["LipPuckerL", "LipPuckerR"],
    "MouthLeft": ["MouthLeft"], "MouthRight": ["MouthRight"],
    "MouthRollUpper": ["LipSuckLt", "LipSuckRt"], "MouthRollLower": ["LipSuckLb", "LipSuckRb"],
    "MouthShrugUpper": ["ChinRaiserT"], "MouthShrugLower": ["ChinRaiserB"],
    "MouthClose": ["LipsToward"],
    "MouthSmileLeft": ["LipCornerPullerL"], "MouthSmileRight": ["LipCornerPullerR"],
    "MouthFrownLeft": ["LipCornerDepressorL"], "MouthFrownRight": ["LipCornerDepressorR"],
    "MouthDimpleLeft": ["DimplerL"], "MouthDimpleRight": ["DimplerR"],
    "MouthUpperUpLeft": ["UpperLipRaiserL"], "MouthUpperUpRight": ["UpperLipRaiserR"],
    "MouthLowerDownLeft": ["LowerLipDepressorL"], "MouthLowerDownRight": ["LowerLipDepressorR"],
    "MouthPressLeft": ["LipPressorL"], "MouthPressRight": ["LipPressorR"],
    "MouthStretchLeft": ["LipStretcherL"], "MouthStretchRight": ["LipStretcherR"],
    "TongueOut": ["TongueOut"],
}

MAPPED = np.array([n in MAP for n in BABBLE])            # True where the Quest Pro can teach


def swap_index():
    """Index that swaps every Left<->Right output (for a mirrored training image). Midline outputs stay."""
    idx = []
    for n in BABBLE:
        o = n.replace("Left", "\0").replace("Right", "Left").replace("\0", "Right")
        idx.append(BABBLE.index(o))
    return np.array(idx)


def labels(qp):
    """qp: dict name -> array (the columns of qp.csv). Returns [N, 45]; unmapped columns are NaN."""
    n = len(next(iter(qp.values())))
    out = np.full((n, len(BABBLE)), np.nan, np.float32)
    for i, name in enumerate(BABBLE):
        if name in MAP:
            out[:, i] = np.mean([qp[q] for q in MAP[name]], axis=0)
    return out


if __name__ == "__main__":
    assert len(BABBLE) == 45 and len(set(BABBLE)) == 45
    s = swap_index()
    assert (s[s] == np.arange(45)).all()
    assert BABBLE[s[BABBLE.index("MouthSmileLeft")]] == "MouthSmileRight"
    assert BABBLE[s[BABBLE.index("JawOpen")]] == "JawOpen"
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import qplog
    missing = [q for qs in MAP.values() for q in qs if q not in qplog.EXPR]
    assert not missing, missing
    print(f"ok: {MAPPED.sum()} of 45 outputs have a Quest Pro teacher; no teacher for: "
          f"{', '.join(n for n in BABBLE if n not in MAP)}")
