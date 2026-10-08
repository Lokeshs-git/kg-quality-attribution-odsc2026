"""Execute the tutorial notebook and save the full, executed copy to local/.

The public notebook at the repository root stays free of outputs for the tutorial.
The executed copy goes to local/kg_diagnosis_adventureworks_full.ipynb, which git ignores.

Run from the repository root:  python tools/run_notebook.py
"""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

src = sys.argv[1] if len(sys.argv) > 1 else "kg_diagnosis_adventureworks.ipynb"
out = sys.argv[2] if len(sys.argv) > 2 else "local/kg_diagnosis_adventureworks_full.ipynb"
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=1500, kernel_name="python3", resources={"metadata": {"path": "."}})
Path(out).parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
try:
    client.execute()
finally:
    nbformat.write(nb, out)
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        print(f"\n===== cell {i}")
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                print(o.text[-3000:])
            elif o.output_type in ("execute_result", "display_data"):
                d = o.get("data", {})
                if "text/plain" in d and "image/png" not in d:
                    print(d["text/plain"][:3000])
                elif "image/png" in d:
                    print("[image]")
            elif o.output_type == "error":
                print("ERROR", o.ename, o.evalue)
                print("\n".join(o.traceback)[-3000:])
    print(f"\nTotal {time.time() - t0:.0f}s")
