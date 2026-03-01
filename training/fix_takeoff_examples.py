"""One-time fixup: replace truncated takeoff examples in train/val.jsonl.

Truncated examples have start_offboard_heartbeat but no offboard_takeoff.
This script replaces them with the correct 2-call sequence: arm + offboard_takeoff.
It also removes disarm/stop_offboard_heartbeat from any landing examples.
"""

import json
import re
import sys
from pathlib import Path


def parse_altitude(user_text: str, assistant_text: str) -> float:
    """Extract altitude from assistant preamble or user prompt, default 1.5m."""
    # Try assistant text first (e.g. "taking off to 2 meters")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|m\b)", assistant_text)
    if m:
        return float(m.group(1))
    # Try user text
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|m\b)", user_text)
    if m:
        return float(m.group(1))
    # Word numbers
    word_map = {"one": 1.0, "two": 2.0, "three": 3.0, "half": 0.5}
    for word, val in word_map.items():
        if word in user_text.lower() or word in assistant_text.lower():
            return val
    return 1.5


def make_arm_call() -> str:
    return (
        '<tool_call>\n'
        '{"name": "call_service", "arguments": {"service": "/dexi/execute_blockly_command", '
        '"service_type": "dexi_interfaces/srv/ExecuteBlocklyCommand", '
        '"args": {"command": "arm", "parameter": 0.0, "timeout": 10.0}}}\n'
        '</tool_call>'
    )


def make_takeoff_call(altitude: float) -> str:
    return (
        '<tool_call>\n'
        '{"name": "call_service", "arguments": {"service": "/dexi/execute_blockly_command", '
        '"service_type": "dexi_interfaces/srv/ExecuteBlocklyCommand", '
        '"args": {"command": "offboard_takeoff", "parameter": '
        f'{altitude}, "timeout": 15.0}}}}}}\n'
        '</tool_call>'
    )


def make_land_call() -> str:
    return (
        '<tool_call>\n'
        '{"name": "call_service", "arguments": {"service": "/dexi/execute_blockly_command", '
        '"service_type": "dexi_interfaces/srv/ExecuteBlocklyCommand", '
        '"args": {"command": "land", "parameter": 0.0, "timeout": 30.0}}}\n'
        '</tool_call>'
    )


def fix_file(path: Path) -> tuple[int, int]:
    """Fix takeoff/landing examples. Returns (takeoff_fixed, landing_fixed)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    fixed_takeoff = 0
    fixed_landing = 0
    out = []

    for line in lines:
        obj = json.loads(line)
        msgs = obj["messages"]
        asst_idx = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        asst = msgs[asst_idx]["content"]
        user_text = ""
        for m in msgs:
            if m["role"] == "user":
                user_text = m["content"]

        if "start_offboard_heartbeat" in asst and "offboard_takeoff" not in asst:
            # Truncated takeoff — rebuild
            altitude = parse_altitude(user_text, asst)
            # Keep preamble text before first <tool_call>
            preamble = asst.split("<tool_call>")[0].rstrip()
            # Clean up preamble to remove references to heartbeat/offboard system
            preamble = re.sub(
                r"(?:I'll |I will )?(?:start (?:the )?offboard (?:system|control),?\s*)?",
                "", preamble, count=1, flags=re.I
            ).strip()
            if not preamble:
                preamble = f"Arming and taking off to {altitude:g} meters."
            else:
                # Capitalize first letter
                preamble = preamble[0].upper() + preamble[1:]

            new_asst = f"{preamble}\n\n{make_arm_call()}\n{make_takeoff_call(altitude)}"
            msgs[asst_idx]["content"] = new_asst
            fixed_takeoff += 1

        elif "stop_offboard_heartbeat" in asst or (
            '"land"' in asst and '"disarm"' in asst
        ):
            # Landing with extra calls — keep just land
            preamble = asst.split("<tool_call>")[0].rstrip()
            new_asst = f"{preamble}\n\n{make_land_call()}"
            msgs[asst_idx]["content"] = new_asst
            fixed_landing += 1

        out.append(json.dumps(obj, ensure_ascii=False))

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return fixed_takeoff, fixed_landing


if __name__ == "__main__":
    base = Path(__file__).parent / "dataset"
    train_path = base / "train.jsonl"
    val_path = base / "val.jsonl"

    t_takeoff, t_landing = fix_file(train_path)
    v_takeoff, v_landing = fix_file(val_path)

    # Verify line counts
    train_lines = len(train_path.read_text().splitlines())
    val_lines = len(val_path.read_text().splitlines())

    print(f"train.jsonl: {t_takeoff} takeoff fixed, {t_landing} landing fixed, {train_lines} lines")
    print(f"val.jsonl:   {v_takeoff} takeoff fixed, {v_landing} landing fixed, {val_lines} lines")

    # Verify no heartbeat references remain
    train_text = train_path.read_text()
    val_text = val_path.read_text()
    hb_train = train_text.count("start_offboard_heartbeat")
    hb_val = val_text.count("start_offboard_heartbeat")
    shb_train = train_text.count("stop_offboard_heartbeat")
    shb_val = val_text.count("stop_offboard_heartbeat")

    print(f"\nVerification:")
    print(f"  start_offboard_heartbeat: train={hb_train}, val={hb_val}")
    print(f"  stop_offboard_heartbeat:  train={shb_train}, val={shb_val}")
    print(f"  offboard_takeoff in train: {train_text.count('offboard_takeoff')}")
    print(f"  offboard_takeoff in val:   {val_text.count('offboard_takeoff')}")

    if hb_train or hb_val or shb_train or shb_val:
        print("\nERROR: heartbeat references still present!")
        sys.exit(1)
    if train_lines != 768 or val_lines != 85:
        print(f"\nERROR: unexpected line counts (expected 768/85)")
        sys.exit(1)
    print("\nAll checks passed!")
