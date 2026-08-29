# ============================================
# Cell 23. Syntax-check historical versions
# of phase_inference_model.py
# ============================================

import subprocess

candidate_commits = [
    "2659775",
    "91123be",
    "a092e1e",
]

for commit in candidate_commits:

    result = subprocess.run(
        [
            "git",
            "-C", str(velocycle_repo),
            "show",
            f"{commit}:velocycle/phase_inference_model.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    source = result.stdout

    try:
        compile(
            source,
            f"{commit}:phase_inference_model.py",
            "exec",
        )
        status = "VALID"
        error = ""

    except SyntaxError as e:
        status = "INVALID"
        error = f"line {e.lineno}: {e.msg}"

    print(f"{commit}: {status}")

    if error:
        print("   ", error)