# A ML Attribution Framework for Measuring Knowledge Graph Quality in GraphRAG

Tutorial materials for ODSC West 2026, October 27 to 29. Presenter: Lokesh Manohar.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lokeshs-git/kg-quality-attribution-odsc2026/blob/main/kg_diagnosis_adventureworks.ipynb)

Knowledge graph defects rarely cause empty answers. They cause plausible wrong ones: a supply-risk list that misses a vendor, or a parts list that includes components from another assembly. This tutorial shows how to measure a graph against a golden question set, find which structural defects cause the wrong answers, and decide which data-quality fix to fund first.

## The method

The notebook runs one loop until the graph meets a benchmark.

1. **Measure.** Answer every question in a golden dataset against the graph and score it by exact match. The gold answers come from SQL on the source tables and never touch the graph.
2. **Diagnose.** Every query leaves a structural trace. The trace becomes 23 features: defect signals such as schema, hierarchy, cardinality and duplicate violations on the query's path, which are zero on valid data, and exposure features that describe the question's shape. A LightGBM classifier with SHAP assigns each wrong answer to the data-quality control that would prevent it.
3. **Fix.** Apply the control with the most attributed wrong answers, using only the schema and the source system, never the gold answers.
4. **Repeat.** Re-measure on held-out questions after every fix.

The graph is built from Microsoft's AdventureWorks sample database: 17 entity types, 22 relationship types, about 57,000 entities and 271,000 relationships, with 1,214 questions across 13 templates. Four templates recurse through the bill of materials.

## Results

On the notebook's graph under test, held-out exact-match accuracy rises from 70.8% to 100% in four rounds, and the attribution chooses the best available fix in three of the four.

That 100% is an artifact of the setup: every fix exactly reverses one kind of simulated defect. Real fixes are partial. The more informative result is the robustness check in `tools/robustness_check.py`, which builds 16 graphs with randomly scaled defect rates and compares how often each method picks the fix that actually recovers the most answers.

| Method for choosing the first fix | Best fix chosen | Answers lost per 1,000 versus the best fix |
| --- | --- | --- |
| SHAP attribution | 9 of 16 | 14.7 |
| Heuristic: control whose signals fire on most wrong answers | 2 of 16 | 57.1 |
| Random choice | about 4 of 16 | n/a |

Full results are in `docs/robustness_results.csv`. The attribution carries real signal, but treat its ranking as a shortlist to verify, not a forecast.

## Running it

**Google Colab (recommended).** Click the badge above and choose Runtime > Run all. The first cell clones this repository and installs DuckDB, LightGBM and SHAP. A full run takes about three minutes on a free CPU runtime and uses under 2 GB of memory. No API keys or GPU are needed.

**Locally.** Python 3.10 or newer:

```bash
git clone https://github.com/Lokeshs-git/kg-quality-attribution-odsc2026.git
cd kg-quality-attribution-odsc2026
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab kg_diagnosis_adventureworks.ipynb
```

After the first run everything is local, and the notebook needs no network. The notebook is committed without outputs, so the results appear as you run it.

Attendee setup instructions are also in `docs/ODSC_West_2026_KG_Attribution_Prerequisites.pdf`.

## Repository layout

| Path | Contents |
| --- | --- |
| `kg_diagnosis_adventureworks.ipynb` | The tutorial notebook, without outputs |
| `prepare_adventureworks.py` | Rebuilds `data/aw` from Microsoft's public AdventureWorks release |
| `data/aw/` | 24 AdventureWorks tables as Parquet (2.7 MB) and Microsoft's license |
| `requirements.txt` | Python dependencies, tested on both minimum and current versions |
| `tools/build_notebook.py` | Source of truth for the notebook: edit cells here, then regenerate |
| `tools/run_notebook.py` | Executes the notebook and saves the full copy to `local/`, which git ignores |
| `tools/robustness_check.py` | The 16-pipeline comparison of SHAP, a heuristic and measured recovery |
| `docs/` | Robustness results and the attendee prerequisites PDF |

To change the notebook, edit `tools/build_notebook.py` and run these from the repository root:

```bash
python tools/build_notebook.py    # regenerates the public notebook, without outputs
python tools/run_notebook.py      # runs it and writes the executed copy to local/
```

The `local/` folder is in `.gitignore`, so executed notebooks never reach the public repository.

## Limits

The question parser is assumed perfect, so the results measure graph quality alone, not how well an LLM turns questions into graph queries. The golden dataset defines quality: defects that no question touches cost nothing here. The pipeline's defects are simulated so that the diagnosis can be checked against known causes, while the fixes are ordinary data-quality procedures, two of which reconcile against a source system. The diagnosis finds only defects it has detectors for; dropped many-to-many links, such as a missing vendor-to-product relationship, leave no trace on a query's path, and the notebook shows the round where that costs it the best choice.

## Using it on your own graph

Replace four things and keep the rest: `build_graph` (your entities, relationships and schema), `TEMPLATES` and `GOLD_SQL` (your questions and an independent source of correct answers), `PROPS` and `CONTROL` (your properties and the controls your team owns), and `FIXES` (your data-quality procedures). Required properties and cardinality rules are learned from the source automatically.

## License

Code is MIT licensed; see `LICENSE`. The AdventureWorks data is Microsoft sample data, redistributed under its own MIT license in `data/aw/LICENSE-AdventureWorks.txt`.
