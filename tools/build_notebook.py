"""Builds kg_diagnosis_adventureworks.ipynb from the cell list below.

Edit the cells here, then run `python tools/build_notebook.py` from the repository
root to regenerate the notebook, and `python tools/run_notebook.py` to execute it
and save its outputs. `python tools/build_notebook.py --exec` runs the code cells
in one namespace as a quick smoke test.
"""

import sys

import nbformat as nbf
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append(("md", s.strip("\n"), {}))


def code(s, form=False):
    # form=True collapses the cell in Colab so only its #@title line shows.
    CELLS.append(("code", s.strip("\n"), {"cellView": "form"} if form else {}))


# ---------------------------------------------------------------------------
md(r"""
# Measuring and Improving Knowledge Graph Quality

A knowledge graph is only as good as the answers it gives. This notebook takes a graph that answers too many questions wrongly, measures how wrong it is against a golden dataset, finds which structural problems cause the failures, fixes the most damaging one, and measures again, until the graph meets a benchmark.

**The loop**

1. **Measure.** Score every question in the golden dataset against the graph. Accuracy is exact-match: a supply-risk list that misses one vendor, or adds a phantom one, is wrong.
2. **Diagnose.** Every query leaves a structural trace. We turn traces into 23 features, train a classifier to predict which questions fail, and use SHAP to assign each failure to the data-quality control that would prevent it.
3. **Fix.** Apply the control with the most attributed failures.
4. **Repeat** until accuracy on held-out questions reaches the benchmark.

The graph is AdventureWorks, Microsoft's sample database for a bicycle manufacturer: 17 entity types and 22 relationship types across manufacturing, purchasing, sales and HR. Everything runs on a free Colab CPU in a few minutes, with no API keys.
""")

md(r"""
## 0. Setup
""")

code(r"""
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO = "https://github.com/Lokeshs-git/kg-quality-attribution-odsc2026.git"

if IN_COLAB and not Path("prepare_adventureworks.py").exists():
    subprocess.run(["git", "clone", "-q", REPO], check=True)
    os.chdir("kg-quality-attribution-odsc2026")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "duckdb>=1.1", "lightgbm>=4.3", "shap>=0.46"], check=True)
""")

code(r"""
import warnings
warnings.filterwarnings("ignore")
import re, time, zlib, random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import duckdb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "#8c8b85", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "font.size": 10,
})
SEED = 42
BENCHMARK = 0.95          # target exact-match accuracy on held-out questions
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. The source system and the golden dataset

The source is the AdventureWorks relational database: 24 tables, about 3 MB as Parquet, shipped in this repository. If the Parquet files are missing, the prep script rebuilds them from Microsoft's public release.

The golden dataset is a set of questions whose correct answers come from SQL on these source tables. It never touches the graph, so it is an independent yardstick.
""")

code(r"""
DATA_DIR = Path("data/aw")
if not (DATA_DIR / "Product.parquet").exists():
    subprocess.run([sys.executable, "prepare_adventureworks.py", "--out", str(DATA_DIR)], check=True)

T = {p.stem: pd.read_parquet(p) for p in sorted(DATA_DIR.glob("*.parquet"))}
con = duckdb.connect()
for name, df in T.items():
    con.register(name, df)
print(f"{len(T)} tables, {sum(len(d) for d in T.values()):,} rows")
""")

code(r'''
def step(rel, direction, target, closure=False, include_self=False):
    return dict(rel=rel, dir=direction, type=target, closure=closure, include_self=include_self)

# Each question template: a seed entity type, a typed path through the graph, and an optional answer filter.
TEMPLATES = {
    "vendor_products": dict(
        seed="Vendor", text="Which products does vendor '{}' supply?",
        path=[step("SUPPLIES", "out", "Product")]),
    "vendor_categories": dict(
        seed="Vendor", text="Which product categories does vendor '{}' supply into?",
        path=[step("SUPPLIES", "out", "Product"), step("IN_SUBCATEGORY", "out", "Subcategory"),
              step("IN_CATEGORY", "out", "Category")]),
    "product_components": dict(
        seed="Product", text="Which components, at any level, go into '{}'?",
        path=[step("HAS_COMPONENT", "out", "Product", closure=True)]),
    "product_part_vendors": dict(
        seed="Product", text="Which vendors supply parts for '{}'?",
        path=[step("HAS_COMPONENT", "out", "Product", closure=True), step("SUPPLIES", "in", "Vendor")]),
    "vendor_exposure": dict(
        seed="Vendor", text="If vendor '{}' stops supplying, which finished products are affected?",
        path=[step("SUPPLIES", "out", "Product"),
              step("HAS_COMPONENT", "in", "Product", closure=True, include_self=True)],
        filter=("FinishedGoodsFlag", 1)),
    "component_locations": dict(
        seed="Product", text="Which inventory locations stock components of '{}'?",
        path=[step("HAS_COMPONENT", "out", "Product", closure=True), step("STOCKED_AT", "out", "Location")]),
    "product_territories": dict(
        seed="Product", text="Which sales territories have ordered '{}'?",
        path=[step("CONTAINS", "in", "SalesOrder"), step("BOOKED_IN", "out", "Territory")]),
    "salesperson_categories": dict(
        seed="SalesPerson", text="Which product categories has '{}' sold?",
        path=[step("SOLD_BY", "in", "SalesOrder"), step("CONTAINS", "out", "Product"),
              step("IN_SUBCATEGORY", "out", "Subcategory"), step("IN_CATEGORY", "out", "Category")]),
    "store_products": dict(
        seed="Store", text="Which products has store '{}' bought?",
        path=[step("BELONGS_TO", "in", "Customer"), step("PLACED", "out", "SalesOrder"),
              step("CONTAINS", "out", "Product")]),
    "store_territory": dict(
        seed="Store", text="Which territory does the salesperson managing '{}' cover?",
        path=[step("MANAGED_BY", "out", "SalesPerson"), step("ASSIGNED_TO", "out", "Territory")]),
    "subcategory_preferred_vendors": dict(
        seed="Subcategory", text="Which preferred vendors supply products in '{}'?",
        path=[step("IN_SUBCATEGORY", "in", "Product"), step("SUPPLIES", "in", "Vendor")],
        filter=("PreferredVendorStatus", 1)),
    "vendor_ship_methods": dict(
        seed="Vendor", text="Which ship methods are used on purchase orders to '{}'?",
        path=[step("ORDERED_FROM", "in", "PurchaseOrder"), step("SHIPPED_VIA", "out", "ShipMethod")]),
    "subcategory_black_products": dict(
        seed="Subcategory", text="Which black products are in '{}'?",
        path=[step("IN_SUBCATEGORY", "in", "Product")], filter=("Color", "Black")),
}

# Gold answers, written independently in SQL against the source tables.
BOM = """bom AS (SELECT ProductAssemblyID a, ComponentID c FROM BillOfMaterials
                 WHERE ProductAssemblyID IS NOT NULL AND EndDate IS NULL)"""
DOWN = BOM + """, down(root, c) AS (SELECT a, c FROM bom UNION SELECT down.root, bom.c FROM down JOIN bom ON bom.a = down.c)"""
GOLD_SQL = {
    "vendor_products": ("Product", "SELECT BusinessEntityID s, ProductID a FROM ProductVendor"),
    "vendor_categories": ("Category", """SELECT pv.BusinessEntityID s, sc.ProductCategoryID a FROM ProductVendor pv
        JOIN Product p USING (ProductID) JOIN ProductSubcategory sc USING (ProductSubcategoryID)"""),
    "product_components": ("Product", f"WITH RECURSIVE {DOWN} SELECT root s, c a FROM down"),
    "product_part_vendors": ("Vendor", f"""WITH RECURSIVE {DOWN} SELECT root s, pv.BusinessEntityID a
        FROM down JOIN ProductVendor pv ON pv.ProductID = down.c"""),
    "vendor_exposure": ("Product", f"""WITH RECURSIVE {BOM}, up(v, p) AS (
            SELECT BusinessEntityID, ProductID FROM ProductVendor
            UNION SELECT up.v, bom.a FROM up JOIN bom ON bom.c = up.p)
        SELECT v s, p a FROM up JOIN Product ON Product.ProductID = up.p WHERE FinishedGoodsFlag = 1"""),
    "component_locations": ("Location", f"""WITH RECURSIVE {DOWN} SELECT root s, i.LocationID a
        FROM down JOIN ProductInventory i ON i.ProductID = down.c"""),
    "product_territories": ("Territory", """SELECT d.ProductID s, h.TerritoryID a
        FROM SalesOrderDetail d JOIN SalesOrderHeader h USING (SalesOrderID)"""),
    "salesperson_categories": ("Category", """SELECT h.SalesPersonID s, sc.ProductCategoryID a
        FROM SalesOrderHeader h JOIN SalesOrderDetail d USING (SalesOrderID) JOIN Product p USING (ProductID)
        JOIN ProductSubcategory sc USING (ProductSubcategoryID) WHERE h.SalesPersonID IS NOT NULL"""),
    "store_products": ("Product", """SELECT c.StoreID s, d.ProductID a FROM Customer c
        JOIN SalesOrderHeader h USING (CustomerID) JOIN SalesOrderDetail d USING (SalesOrderID)
        WHERE c.StoreID IS NOT NULL"""),
    "store_territory": ("Territory", """SELECT s.BusinessEntityID s, sp.TerritoryID a FROM Store s
        JOIN SalesPerson sp ON s.SalesPersonID = sp.BusinessEntityID WHERE sp.TerritoryID IS NOT NULL"""),
    "subcategory_preferred_vendors": ("Vendor", """SELECT p.ProductSubcategoryID s, pv.BusinessEntityID a
        FROM Product p JOIN ProductVendor pv USING (ProductID) JOIN Vendor v ON v.BusinessEntityID = pv.BusinessEntityID
        WHERE v.PreferredVendorStatus = 1 AND p.ProductSubcategoryID IS NOT NULL"""),
    "vendor_ship_methods": ("ShipMethod", "SELECT VendorID s, ShipMethodID a FROM PurchaseOrderHeader"),
    "subcategory_black_products": ("Product", """SELECT ProductSubcategoryID s, ProductID a FROM Product
        WHERE Color = 'Black' AND ProductSubcategoryID IS NOT NULL"""),
}
''')

code(r"""
def nid(t, i):
    return f"{t}:{int(i)}"

def full_name(df):
    return dict(zip(df.BusinessEntityID, df.FirstName.fillna("") + " " + df.LastName.fillna("")))

PERSON = full_name(T["Person"])
names_of = {
    "Vendor": dict(zip(T["Vendor"].BusinessEntityID, T["Vendor"].Name)),
    "Product": dict(zip(T["Product"].ProductID, T["Product"].Name)),
    "Store": dict(zip(T["Store"].BusinessEntityID, T["Store"].Name)),
    "SalesPerson": PERSON,
    "Subcategory": dict(zip(T["ProductSubcategory"].ProductSubcategoryID, T["ProductSubcategory"].Name)),
}

def make_questions(per_template=150, seed=SEED):
    rng = random.Random(seed)
    qs = []
    for tname, tpl in TEMPLATES.items():
        ans_type, sql = GOLD_SQL[tname]
        pairs = con.execute(sql).df().dropna()
        gold = pairs.groupby("s").a.apply(lambda s: frozenset(nid(ans_type, x) for x in s))
        nm = names_of[tpl["seed"]]
        counts = Counter(nm.values())
        seeds = [(sid, nm[sid], ans) for sid, ans in gold.items()
                 if ans and isinstance(nm.get(sid), str) and counts[nm[sid]] == 1]   # unique names only
        rng.shuffle(seeds)
        for sid, name, ans in sorted(seeds[:per_template]):
            qs.append(dict(qid=len(qs), template=tname, seed_name=name.strip(),
                           question=tpl["text"].format(name.strip()), gold=ans))
    return qs

QUESTIONS = make_questions()
qdf = pd.DataFrame(QUESTIONS)
# Diagnose on the dev half, report the benchmark on the held-out test half.
dev_ids, test_ids = train_test_split(qdf.qid, test_size=0.5, stratify=qdf.template, random_state=SEED)
DEV, TEST = set(dev_ids), set(test_ids)
print(f"{len(QUESTIONS)} questions: {len(DEV)} for diagnosis, {len(TEST)} held out for the benchmark\n")
print(qdf.iloc[[0, 300, 700]].question.to_string(index=False))
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. The graph under test

Each ID column in the source becomes an entity type and each foreign key a typed, directed relationship. `SCHEMA` records every allowed (source type, relationship, target type) triple.

We build the graph twice. `G_SOURCE` is a faithful copy of the source system, which the fixes later reconcile against, the way a team reconciles a graph with its system of record. `G` is the graph under test: the same build passed through a simulated production pipeline that, like real pipelines, introduces defects along the way. Which defects, and how many, stays hidden until section 8. The diagnosis has to find them.
""")

code(r"""
PROPS = {
    "Product": ["ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "ListPrice",
                "StandardCost", "Size", "Weight", "ProductLine", "Class", "Style"],
    "Subcategory": [], "Category": [], "ProductModel": [],
    "Vendor": ["AccountNumber", "CreditRating", "PreferredVendorStatus", "ActiveFlag"],
    "Location": ["CostRate", "Availability"],
    "PurchaseOrder": ["Status", "OrderDate", "TotalDue"],
    "ShipMethod": ["ShipBase", "ShipRate"],
    "Employee": ["JobTitle", "HireDate", "SalariedFlag"],
    "Department": ["GroupName"],
    "Customer": ["AccountNumber"],
    "Store": [],
    "SalesPerson": ["SalesQuota", "Bonus", "CommissionPct"],
    "Territory": ["Group", "CountryRegionCode"],
    "SalesOrder": ["OrderDate", "Status", "OnlineOrderFlag", "SubTotal", "TotalDue"],
    "SpecialOffer": ["DiscountPct", "Type", "Category"],
    "Country": [],
}

def build_graph(T):
    G = nx.MultiDiGraph()
    schema = set()

    def nodes(df, t, id_col, name):
        names = name(df) if callable(name) else df[name]
        props = PROPS[t]
        recs = df[props].to_dict("records") if props else [{}] * len(df)
        for i, nm, rec in zip(df[id_col], names, recs):
            attrs = {k: v for k, v in rec.items() if pd.notna(v)}
            if isinstance(nm, str) and nm.strip():
                attrs["name"] = nm.strip()
            G.add_node(nid(t, i), type=t, **attrs)

    def edges(df, st, sc, rel, dt, dc, props=()):
        df = df.dropna(subset=[sc, dc])
        for rec in df[[sc, dc, *props]].itertuples(index=False):
            u, v = nid(st, rec[0]), nid(dt, rec[1])
            if u in G and v in G:
                G.add_edge(u, v, key=rel, rel=rel,
                           **{p: rec[2 + j] for j, p in enumerate(props) if pd.notna(rec[2 + j])})
        schema.add((st, rel, dt))

    t = T
    nodes(t["Product"], "Product", "ProductID", "Name")
    nodes(t["ProductSubcategory"], "Subcategory", "ProductSubcategoryID", "Name")
    nodes(t["ProductCategory"], "Category", "ProductCategoryID", "Name")
    nodes(t["ProductModel"], "ProductModel", "ProductModelID", "Name")
    nodes(t["Vendor"], "Vendor", "BusinessEntityID", "Name")
    nodes(t["Location"], "Location", "LocationID", "Name")
    nodes(t["PurchaseOrderHeader"], "PurchaseOrder", "PurchaseOrderID",
          lambda d: "PO" + d.PurchaseOrderID.astype(int).astype(str))
    nodes(t["ShipMethod"], "ShipMethod", "ShipMethodID", "Name")
    nodes(t["Employee"], "Employee", "BusinessEntityID", lambda d: d.BusinessEntityID.map(PERSON))
    nodes(t["Department"], "Department", "DepartmentID", "Name")
    stores = t["Store"]
    store_name = dict(zip(stores.BusinessEntityID, stores.Name))
    cust = t["Customer"]
    nodes(cust, "Customer", "CustomerID",
          lambda d: d.PersonID.map(PERSON).fillna(d.StoreID.map(store_name)))
    nodes(stores, "Store", "BusinessEntityID", "Name")
    nodes(t["SalesPerson"], "SalesPerson", "BusinessEntityID", lambda d: d.BusinessEntityID.map(PERSON))
    nodes(t["SalesTerritory"], "Territory", "TerritoryID", "Name")
    nodes(t["SalesOrderHeader"], "SalesOrder", "SalesOrderID",
          lambda d: "SO" + d.SalesOrderID.astype(int).astype(str))
    nodes(t["SpecialOffer"], "SpecialOffer", "SpecialOfferID", "Description")
    used = set(t["SalesTerritory"].CountryRegionCode)
    countries = t["CountryRegion"][t["CountryRegion"].CountryRegionCode.isin(used)].copy()
    countries["cid"] = range(1, len(countries) + 1)
    cmap = dict(zip(countries.CountryRegionCode, countries.cid))
    nodes(countries, "Country", "cid", "Name")

    p = t["Product"]
    edges(p, "Product", "ProductID", "IN_SUBCATEGORY", "Subcategory", "ProductSubcategoryID")
    edges(t["ProductSubcategory"], "Subcategory", "ProductSubcategoryID", "IN_CATEGORY", "Category", "ProductCategoryID")
    edges(p, "Product", "ProductID", "OF_MODEL", "ProductModel", "ProductModelID")
    bom = t["BillOfMaterials"]
    bom = bom[bom.ProductAssemblyID.notna() & bom.EndDate.isna()]   # current bill of materials
    edges(bom, "Product", "ProductAssemblyID", "HAS_COMPONENT", "Product", "ComponentID", ["PerAssemblyQty", "BOMLevel"])
    edges(t["ProductInventory"], "Product", "ProductID", "STOCKED_AT", "Location", "LocationID", ["Quantity"])
    edges(t["ProductVendor"], "Vendor", "BusinessEntityID", "SUPPLIES", "Product", "ProductID",
          ["AverageLeadTime", "StandardPrice"])
    po = t["PurchaseOrderHeader"]
    edges(po, "PurchaseOrder", "PurchaseOrderID", "ORDERED_FROM", "Vendor", "VendorID")
    edges(po, "PurchaseOrder", "PurchaseOrderID", "SHIPPED_VIA", "ShipMethod", "ShipMethodID")
    edges(po, "PurchaseOrder", "PurchaseOrderID", "CREATED_BY", "Employee", "EmployeeID")
    edges(t["PurchaseOrderDetail"], "PurchaseOrder", "PurchaseOrderID", "INCLUDES", "Product", "ProductID", ["OrderQty"])
    edh = t["EmployeeDepartmentHistory"]
    edges(edh[edh.EndDate.isna()], "Employee", "BusinessEntityID", "WORKS_IN", "Department", "DepartmentID")
    edges(cust, "Customer", "CustomerID", "BELONGS_TO", "Store", "StoreID")
    edges(cust, "Customer", "CustomerID", "LOCATED_IN", "Territory", "TerritoryID")
    edges(stores, "Store", "BusinessEntityID", "MANAGED_BY", "SalesPerson", "SalesPersonID")
    sp = t["SalesPerson"]
    edges(sp, "SalesPerson", "BusinessEntityID", "ASSIGNED_TO", "Territory", "TerritoryID")
    edges(sp, "SalesPerson", "BusinessEntityID", "IS_EMPLOYEE", "Employee", "BusinessEntityID")
    soh = t["SalesOrderHeader"]
    edges(soh, "Customer", "CustomerID", "PLACED", "SalesOrder", "SalesOrderID")
    edges(soh, "SalesOrder", "SalesOrderID", "SOLD_BY", "SalesPerson", "SalesPersonID")
    edges(soh, "SalesOrder", "SalesOrderID", "BOOKED_IN", "Territory", "TerritoryID")
    edges(soh, "SalesOrder", "SalesOrderID", "SHIPPED_VIA", "ShipMethod", "ShipMethodID")
    edges(t["SalesOrderDetail"], "SalesOrder", "SalesOrderID", "CONTAINS", "Product", "ProductID",
          ["OrderQty", "UnitPrice", "UnitPriceDiscount"])
    offers = t["SalesOrderDetail"].query("SpecialOfferID != 1")[["SalesOrderID", "SpecialOfferID"]].drop_duplicates()
    edges(offers, "SalesOrder", "SalesOrderID", "USED_OFFER", "SpecialOffer", "SpecialOfferID")
    terr = t["SalesTerritory"].assign(cid=lambda d: d.CountryRegionCode.map(cmap))
    edges(terr, "Territory", "TerritoryID", "IN_COUNTRY", "Country", "cid")
    return G, schema

G_SOURCE, SCHEMA = build_graph(T)
TYPE_OF = nx.get_node_attributes(G_SOURCE, "type")
print(f"{G_SOURCE.number_of_nodes():,} entities, {G_SOURCE.number_of_edges():,} relationships, "
      f"{len(set(TYPE_OF.values()))} entity types, {len({r for _, r, _ in SCHEMA})} relationship types")
""")

code(r"""
#@title Simulated production pipeline (its defects are revealed in section 8)
def _u01(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) / 2**32

def _variant(name, seed, n):
    # Duplicate records often differ only in case or spacing.
    r = _u01(seed, "variant", n)
    return name if r < 0.5 else (name.upper() if r < 0.75 else name.replace(" ", "  ", 1))

def run_pipeline(G, rates, seed=0):
    H = G.copy()
    log = defaultdict(int)
    if rates.get("attribute"):
        for n, d in H.nodes(data=True):
            for p in [p for p in d if p != "type"]:
                if _u01(seed, "attr", n, p) < rates["attribute"]:
                    del d[p]
                    log["properties dropped"] += 1
    if rates.get("duplicate"):
        for n in [n for n in H.nodes if _u01(seed, "dup", n) < rates["duplicate"]]:
            clone = n + "#dup"
            attrs = dict(H.nodes[n])
            if "name" in attrs:
                attrs["name"] = _variant(attrs["name"], seed, n)
            H.add_node(clone, **attrs)
            moves = list(H.out_edges(n, keys=True, data=True)) + list(H.in_edges(n, keys=True, data=True))
            for u, v, k, d in moves:
                if u != v and _u01(seed, "dup-edge", u, v, k) < 0.5:
                    H.remove_edge(u, v, key=k)
                    H.add_edge(clone if u == n else u, clone if v == n else v, key=k, **d)
            log["entities duplicated"] += 1
    for op, label in (("flip", "edges reversed"), ("delete", "edges dropped")):
        if rates.get(op):
            for u, v, k, d in [e for e in H.edges(keys=True, data=True) if _u01(seed, op, *e[:3]) < rates[op]]:
                H.remove_edge(u, v, key=k)
                if op == "flip":
                    H.add_edge(v, u, key=k, **d)
                log[label] += 1
    return H, dict(log)

PIPELINE_RATES = dict(attribute=0.04, duplicate=0.004, flip=0.002, delete=0.008)
G, PIPELINE_LOG = run_pipeline(G_SOURCE, PIPELINE_RATES, seed=7)
print(f"Graph under test: {G.number_of_nodes():,} entities, {G.number_of_edges():,} relationships")
""", form=True)

# ---------------------------------------------------------------------------
md(r"""
## 3. Measure: how good is this graph?

The query engine answers each question the way a structured GraphRAG retriever would once the question is parsed: resolve the seed entity by name, follow the typed path, apply the filter. Parsing is assumed perfect, so every wrong answer is the graph's fault. That keeps the measurement deterministic, with no LLM randomness to separate from graph quality.

Along the way, the engine records a structural trace of the query. Section 4 explains the features in that trace; here we only use the answers.
""")

code(r"""
def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower()) if isinstance(s, str) else None

SOURCE_KEYS = set(G_SOURCE.nodes)

# Constraints learned from the source system: required properties and relationship cardinality.
_by_type = defaultdict(list)
for _n, _d in G_SOURCE.nodes(data=True):
    _by_type[_d["type"]].append(_n)
REQUIRED = {t: [p for p in PROPS[t] + ["name"] if all(p in G_SOURCE.nodes[n] for n in ns)]
            for t, ns in _by_type.items()}
_out_deg, _in_deg = Counter(), Counter()
for _u, _v, _k in G_SOURCE.edges(keys=True):
    _out_deg[(_u, _k)] += 1
    _in_deg[(_v, _k)] += 1
CARD = defaultdict(list)          # type -> [(rel, direction, exactly_one)]
for st, rel, dt in SCHEMA:
    for t, direction, cnt in ((st, "out", _out_deg), (dt, "in", _in_deg)):
        deg = [cnt[(n, rel)] for n in _by_type[t]]
        if max(deg) == 1:
            CARD[t].append((rel, direction, min(deg) == 1))


class Index:
    # Adjacency by node and relationship in both directions, name lookups, and degree baselines.
    def __init__(self, G):
        self.G, self.out, self.inn = G, {}, {}
        for u, v, k in G.edges(keys=True):
            self.out.setdefault(u, {}).setdefault(k, []).append(v)
            self.inn.setdefault(v, {}).setdefault(k, []).append(u)
        self.ntype = {n: d["type"] for n, d in G.nodes(data=True)}
        self.names, self.norms = defaultdict(list), Counter()
        for n, d in G.nodes(data=True):
            if "name" in d:
                self.names[(d["type"], d["name"])].append(n)
                self.norms[(d["type"], norm(d["name"]))] += 1
        self.deg = {n: self._deg(n) for n in G.nodes}
        by_t = defaultdict(list)
        for n, t in self.ntype.items():
            by_t[t].append(self.deg[n])
        self.median_deg = {t: max(float(np.median(v)), 1.0) for t, v in by_t.items()}

    def _deg(self, n):
        return sum(map(len, self.out.get(n, {}).values())) + sum(map(len, self.inn.get(n, {}).values()))

    def order_mismatch(self, n):
        # A sales order's line items must add up to its subtotal; a missing or misdirected line breaks that.
        sub = self.G.nodes[n].get("SubTotal")
        if self.ntype[n] != "SalesOrder" or sub is None:
            return False
        E = self.G.edges
        tot = sum(E[(n, p, "CONTAINS")].get("OrderQty", 0) * E[(n, p, "CONTAINS")].get("UnitPrice", 0)
                  * (1 - E[(n, p, "CONTAINS")].get("UnitPriceDiscount", 0))
                  for p in self.out.get(n, {}).get("CONTAINS", ()))
        return abs(tot - sub) > 0.005 * max(sub, 1)

    def card_violations(self, n):
        t, bad = self.ntype[n], 0
        for rel, direction, exactly_one in CARD[t]:
            c = len((self.out if direction == "out" else self.inn).get(n, {}).get(rel, ()))
            bad += c > 1 or (exactly_one and c == 0)
        return bad
""")

code(r"""
# Defect signals: constraint violations on the query's path, zero on a valid graph.
DEFECTS = [
    "seed_missing", "seed_ambiguity", "dup_exact", "dup_fuzzy", "unknown_keys",        # entity resolution
    "schema_violations", "hierarchy_violations", "bom_cycles",                          # schema and hierarchy
    "hops_missing", "cardinality_violations", "total_mismatch",                         # referential integrity
    "required_missing", "filter_missing",                                               # completeness
]
# Exposure: how much graph a question touches and how it is shaped. Nonzero on a valid graph too.
EXPOSURE = ["seed_degree", "seed_degree_ratio", "dead_ends", "min_fanout", "low_degree_entities",
            "template_id", "steps", "recursive", "has_filter", "gold_size"]
STRUCTURAL = DEFECTS + EXPOSURE[:5]
FEATURES = DEFECTS + EXPOSURE
TEMPLATE_ID = {t: i for i, t in enumerate(TEMPLATES)}


def run_query(ix, q):
    tpl = TEMPLATES[q["template"]]
    f = dict.fromkeys(STRUCTURAL, 0.0)
    f.update(template_id=TEMPLATE_ID[q["template"]], steps=len(tpl["path"]),
             recursive=float(any(s["closure"] for s in tpl["path"])),
             has_filter=float("filter" in tpl), gold_size=np.log1p(len(q["gold"])))
    cands = ix.names.get((tpl["seed"], q["seed_name"]), [])
    f["seed_missing"], f["seed_ambiguity"] = float(not cands), float(max(len(cands) - 1, 0))
    if not cands:
        f["hops_missing"] = 1.0
        return frozenset(), f
    seed = max(cands, key=lambda n: (ix.deg[n], n))      # an entity linker picks the best-connected match
    f["seed_degree"] = np.log1p(ix.deg[seed])
    f["seed_degree_ratio"] = ix.deg[seed] / ix.median_deg[ix.ntype[seed]]
    E, N = ix.G.edges, ix.G.nodes

    frontier, evidence, intermediates = {seed}, {seed}, set()
    dead = done = viol = hier = cycles = 0
    fanouts = []
    for i, st in enumerate(tpl["path"]):
        down = st["dir"] == "out"
        fwd, bwd = (ix.out, ix.inn) if down else (ix.inn, ix.out)
        rel, want = st["rel"], st["type"]
        if i > 0:
            intermediates |= frontier

        def expand(n):
            nonlocal viol
            for m in fwd.get(n, {}).get(rel, ()):
                viol += ((ix.ntype[n], rel, ix.ntype[m]) if down else (ix.ntype[m], rel, ix.ntype[n])) not in SCHEMA
            for m in bwd.get(n, {}).get(rel, ()):
                viol += ((ix.ntype[m], rel, ix.ntype[n]) if down else (ix.ntype[n], rel, ix.ntype[m])) not in SCHEMA
            return [m for m in fwd.get(n, {}).get(rel, ()) if ix.ntype[m] == want]

        if st["closure"]:
            # Bill-of-materials walk: levels must rise going down and fall going up.
            start = set(frontier)
            level = {n: (-1 if down else 99) for n in frontier}
            reached, stack = set(), list(frontier)
            while stack:
                n = stack.pop()
                for m in expand(n):
                    L = E[(n, m, rel) if down else (m, n, rel)].get("BOMLevel")
                    if L is not None and (L <= level[n] if down else L >= level[n]):
                        hier += 1
                    cycles += m in start
                    if m not in reached:
                        reached.add(m)
                        level[m] = L if L is not None else level[n]
                        stack.append(m)
            new = reached | (frontier if st["include_self"] else set())
        else:
            new, n_edges = set(), 0
            for n in frontier:
                nbrs = expand(n)
                dead += not nbrs
                n_edges += len(nbrs)
                new.update(nbrs)
            fanouts.append(n_edges / len(frontier))
        frontier = new
        evidence |= new
        if not frontier:
            break
        done += 1

    answers = frontier
    if "filter" in tpl:
        prop, val = tpl["filter"]
        f["filter_missing"] = np.log1p(sum(prop not in N[n] for n in answers if prop in REQUIRED[ix.ntype[n]]))
        answers = {n for n in answers if N[n].get(prop) == val}

    f.update(
        dup_exact=np.log1p(sum(len(ix.names.get((ix.ntype[n], N[n].get("name")), ())) > 1 for n in evidence)),
        dup_fuzzy=np.log1p(sum(ix.norms[(ix.ntype[n], norm(N[n].get("name")))] >
                               len(ix.names.get((ix.ntype[n], N[n].get("name")), ())) for n in evidence if "name" in N[n])),
        unknown_keys=np.log1p(sum(n not in SOURCE_KEYS for n in evidence)),
        hops_missing=1 - done / len(tpl["path"]),
        dead_ends=np.log1p(dead),
        min_fanout=min(fanouts) if fanouts else 0.0,
        low_degree_entities=np.log1p(sum(ix.deg[n] < 0.25 * ix.median_deg[ix.ntype[n]] for n in intermediates)),
        schema_violations=np.log1p(viol),
        hierarchy_violations=np.log1p(hier),
        bom_cycles=np.log1p(cycles),
        cardinality_violations=np.log1p(sum(ix.card_violations(n) for n in evidence)),
        total_mismatch=np.log1p(sum(ix.order_mismatch(n) for n in evidence)),
        required_missing=np.log1p(sum(p not in N[n] for n in evidence for p in REQUIRED[ix.ntype[n]])),
    )
    return frozenset(answers), f


def score(answer, gold):
    tp = len(answer & gold)
    p = tp / len(answer) if answer else 0.0
    r = tp / len(gold)
    f1 = 2 * p * r / (p + r) if tp else 0.0
    kind = ("correct" if answer == gold else "empty" if not answer
            else "incomplete" if answer < gold else "contaminated")
    return f1, kind


def evaluate(G):
    ix = Index(G)
    rows = []
    for q in QUESTIONS:
        ans, f = run_query(ix, q)
        f1, kind = score(ans, q["gold"])
        rows.append(dict(qid=q["qid"], template=q["template"], split="dev" if q["qid"] in DEV else "test",
                         f1=f1, kind=kind, wrong=int(kind != "correct"), **f))
    return pd.DataFrame(rows)


def accuracy(res, split="test"):
    return 1 - res[res.split == split].wrong.mean()
""")

code(r"""
# Sanity check: on a faithful copy of the source, the graph must reproduce the golden dataset exactly.
res_source = evaluate(G_SOURCE)
print(f"Faithful copy of the source: {accuracy(res_source, 'dev'):.1%} dev, {accuracy(res_source):.1%} test")
assert res_source.wrong.sum() == 0

RES0 = evaluate(G)
print(f"\nGraph under test: {accuracy(RES0):.1%} exact-match accuracy on held-out questions "
      f"(benchmark {BENCHMARK:.0%}), mean F1 {RES0[RES0.split == 'test'].f1.mean():.3f}")
by_t = (RES0.groupby("template").agg(accuracy=("wrong", lambda s: 1 - s.mean()),
                                     empty=("kind", lambda s: (s == "empty").mean()),
                                     incomplete=("kind", lambda s: (s == "incomplete").mean()),
                                     contaminated=("kind", lambda s: (s == "contaminated").mean()))
        .sort_values("accuracy"))
by_t.round(2)
""")

md(r"""
Mean F1 looks healthy while exact-match accuracy does not. Most wrong answers are incomplete or contaminated rather than empty: they look plausible, and nothing about them signals a problem. That is why a golden dataset is needed at all.

### A graph-wide health report

Before diagnosing individual questions, the same constraints the features use can be checked across the whole graph. The report shows that defects exist; it cannot say which ones cost answers. That is the diagnosis's job.
""")

code(r"""
def bom_consistent(H, a, b, L):
    # An assembly->component edge a->b at level L fits if b's own components sit at L+1 and a's parents
    # at L-1. The edge between a and b, in either direction, is left out of the check.
    ok_b = all(H.edges[b, c, "HAS_COMPONENT"].get("BOMLevel") == L + 1
               for _, c, k in H.out_edges(b, keys=True) if k == "HAS_COMPONENT" and c != a)
    ok_a = all(H.edges[p, a, "HAS_COMPONENT"].get("BOMLevel") == L - 1
               for p, _, k in H.in_edges(a, keys=True) if k == "HAS_COMPONENT" and p != b)
    return ok_a and ok_b

def health(H):
    ix = Index(H)
    n, m = H.number_of_nodes(), H.number_of_edges()
    TY = ix.ntype
    bom = [(u, v, d.get("BOMLevel")) for u, v, k, d in H.edges(keys=True, data=True) if k == "HAS_COMPONENT"]
    return {
        "entities missing a required property": sum(any(p not in d for p in REQUIRED[d["type"]]) for _, d in H.nodes(data=True)) / n,
        "entities sharing a normalized name": sum(c for c in ix.norms.values() if c > 1) / n,
        "entities with keys unknown to the source": sum(x not in SOURCE_KEYS for x in H.nodes) / n,
        "edges violating the schema": sum((TY[u], k, TY[v]) not in SCHEMA for u, v, k in H.edges(keys=True)) / m,
        "BOM edges with inconsistent levels": sum(not bom_consistent(H, u, v, L) for u, v, L in bom if L is not None) / max(len(bom), 1),
        "entities violating cardinality": sum(ix.card_violations(x) > 0 for x in H.nodes) / n,
        "sales orders whose lines miss their subtotal": sum(ix.order_mismatch(x) for x in H.nodes) / n,
    }

HEALTH = {"source copy": health(G_SOURCE), "graph under test": health(G)}
(pd.DataFrame(HEALTH) * 100).round(2).rename(columns=lambda c: c + " (%)")
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Diagnose: which structural problems cause the wrong answers?

Every query records 23 features, all computable in production without the gold answer. They come in two kinds.

**Defect signals** count constraint violations on the query's path. Each is zero on a valid graph, and each points at the control that prevents it.

| Control | Feature | What it counts on the query's path |
| --- | --- | --- |
| Deduplication | `seed_ambiguity` | Extra entities sharing the seed's name |
| | `dup_exact`, `dup_fuzzy` | Touched entities with an exact or near-duplicate name twin (case and spacing ignored) |
| | `unknown_keys` | Touched entities whose key does not exist in the source system |
| Schema and hierarchy | `schema_violations` | Edges whose source and target types break the schema |
| | `hierarchy_violations` | Bill-of-materials edges whose level runs the wrong way |
| | `bom_cycles` | Bill-of-materials walks that loop back to where they started |
| Referential integrity | `hops_missing` | Share of the path's steps that reached nothing |
| | `cardinality_violations` | Entities with too many or too few of a relationship (an order needs exactly one territory) |
| | `total_mismatch` | Sales orders whose line items no longer add up to the order's subtotal |
| Completeness | `seed_missing` | The seed's name is missing, so it cannot be found |
| | `required_missing` | Required properties missing on touched entities |
| | `filter_missing` | Candidate answers missing the property the filter needs |

**Exposure features** describe how much graph a question touches and how it is shaped: `seed_degree`, `seed_degree_ratio`, `dead_ends`, `min_fanout` and `low_degree_entities` from the trace, plus `template_id`, `steps`, `recursive`, `has_filter` and `gold_size` from the question. A question that walks a large bill of materials has more ways to fail than a one-hop lookup, and the model needs to know that. But exposure is not damage, so blame that lands on it is reported as question exposure and never sent to a fix.

The split is checked, not assumed: the next cell counts how often each feature is nonzero on the faithful source copy.

Two design rules keep the diagnosis honest. Defect signals are counts, not averages over the touched subgraph: averages shift whenever the answer changes, and a model built on them predicts wrong answers from wrong answers. And a feature counts toward a fix only if it is zero on valid data. An earlier version credited seed degree and fanout to referential integrity, and the loop's first fix recovered almost nothing.

The classifier learns from variation across questions: which ones failed and what was different about their paths. SHAP then splits each wrong answer's blame among the features that pushed it toward wrong. Blame on defect signals rolls up to the control that would prevent the defect; blame on exposure stays with the question.
""")

code(r"""
nonzero = pd.DataFrame({
    "valid graph": (res_source[STRUCTURAL] != 0).mean(),
    "graph under test": (RES0[STRUCTURAL] != 0).mean(),
}).mul(100).round(1)
nonzero.insert(0, "kind", ["defect signal"] * len(DEFECTS) + ["exposure"] * (len(STRUCTURAL) - len(DEFECTS)))
nonzero.rename(columns={"valid graph": "% nonzero, valid graph", "graph under test": "% nonzero, graph under test"})
""")

code(r"""
CONTROL = {
    "seed_missing": "Completeness", "required_missing": "Completeness", "filter_missing": "Completeness",
    "seed_ambiguity": "Deduplication", "dup_exact": "Deduplication", "dup_fuzzy": "Deduplication",
    "unknown_keys": "Deduplication",
    "schema_violations": "Schema and hierarchy", "hierarchy_violations": "Schema and hierarchy",
    "bom_cycles": "Schema and hierarchy",
    "hops_missing": "Referential integrity", "cardinality_violations": "Referential integrity",
    "total_mismatch": "Referential integrity",
    **{c: "Question exposure" for c in EXPOSURE},
}
PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15,
              min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=SEED, verbose=-1)

def diagnose(res):
    d = res[res.split == "dev"]
    X, y = d[FEATURES], d.wrong
    auc = None
    if y.sum() >= 10:
        oof = np.zeros(len(X))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
            oof[te] = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y, oof)
    model = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    # Each wrong answer carries one unit of blame, split across the features that pushed it toward wrong.
    P = np.clip(sv[(y == 1).values], 0, None)
    P = P[P.sum(axis=1) > 0]
    P = P / P.sum(axis=1, keepdims=True)
    share = pd.Series(P.mean(axis=0), index=FEATURES)
    wrong_per_1000 = y.mean() * 1000
    table = (share.groupby(share.index.map(CONTROL)).sum().sort_values(ascending=False)
             .to_frame("blame_share")
             .assign(attributed_wrong_per_1000=lambda t: (t.blame_share * wrong_per_1000).round(1)))
    return dict(model=model, X=X, sv=sv, auc=auc, share=share, table=table)

D0 = diagnose(RES0)
print(f"Classifier AUC (5-fold, out of fold): {D0['auc']:.3f}")
D0["table"].assign(blame_share=lambda t: (t.blame_share * 100).round(1).astype(str) + "%")
""")

code(r"""
shap.summary_plot(D0["sv"], D0["X"], show=False, max_display=14, plot_size=(8, 5))
plt.title("What pushes a question toward a wrong answer", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Fix the top problem and measure again

Each control is a real data-quality procedure. None of them looks at the golden answers or at the pipeline's hidden defect log. They use only the schema and the source system of record.

| Control | Procedure |
| --- | --- |
| Schema and hierarchy | Reverse edges whose types only fit the schema backwards; reverse bill-of-materials edges whose levels only fit backwards |
| Deduplication | Group entities by type and normalized name; merge any entity whose key is unknown to the source into its known twin |
| Completeness | Re-sync missing properties from the source for every entity with a known key |
| Referential integrity | Reconcile foreign keys: add relationships present in the source but missing from the graph |

In practice each of these is weeks of engineering, which is why the order matters. Here they run in seconds.
""")

code(r"""
def fix_schema_and_hierarchy(H):
    n = 0
    TY = nx.get_node_attributes(H, "type")
    for u, v, k, d in list(H.edges(keys=True, data=True)):
        if (TY[u], k, TY[v]) not in SCHEMA and (TY[v], k, TY[u]) in SCHEMA:
            H.remove_edge(u, v, key=k); H.add_edge(v, u, key=k, **d); n += 1
    for _ in range(3):                      # a few passes, since one reversal can unblock another
        changed = 0
        for u, v, k, d in list(H.edges(keys=True, data=True)):
            L = d.get("BOMLevel")
            if k == "HAS_COMPONENT" and L is not None and not bom_consistent(H, u, v, L) \
                    and bom_consistent(H, v, u, L):
                H.remove_edge(u, v, key=k); H.add_edge(v, u, key=k, **d); changed += 1
        n += changed
        if not changed:
            break
    return f"{n:,} edges reversed"

def fix_deduplication(H):
    groups = defaultdict(list)
    for x, d in H.nodes(data=True):
        if "name" in d:
            groups[(d["type"], norm(d["name"]))].append(x)
    merged = 0
    for members in groups.values():
        known = [x for x in members if x in SOURCE_KEYS]
        for dup in [x for x in members if x not in SOURCE_KEYS]:
            if not known:
                continue
            nb = set(H.successors(dup)) | set(H.predecessors(dup))
            target = max(known, key=lambda x: (len(nb & (set(H.successors(x)) | set(H.predecessors(x)))), x))
            for _, v, k, d in list(H.out_edges(dup, keys=True, data=True)):
                H.add_edge(target, v, key=k, **d)
            for u, _, k, d in list(H.in_edges(dup, keys=True, data=True)):
                H.add_edge(u, target, key=k, **d)
            H.remove_node(dup); merged += 1
    return f"{merged:,} duplicate entities merged"

def fix_completeness(H):
    restored = 0
    for x, d in H.nodes(data=True):
        if x in SOURCE_KEYS:
            for p, val in G_SOURCE.nodes[x].items():
                if p not in d:
                    d[p] = val; restored += 1
    return f"{restored:,} properties restored"

def fix_referential_integrity(H):
    added = 0
    for u, v, k, d in G_SOURCE.edges(keys=True, data=True):
        if u in H and v in H and not H.has_edge(u, v, k) and not H.has_edge(v, u, k):
            H.add_edge(u, v, key=k, **d); added += 1
    return f"{added:,} relationships restored"

FIXES = {"Schema and hierarchy": fix_schema_and_hierarchy, "Deduplication": fix_deduplication,
         "Completeness": fix_completeness, "Referential integrity": fix_referential_integrity}

def next_fix(diag, done):
    return next(c for c in diag["table"].index if c in FIXES and c not in done)
""")

code(r"""
choice = next_fix(D0, [])
G1 = G.copy()
print(f"Top attributed control: {choice} -> {FIXES[choice](G1)}")
RES1 = evaluate(G1)
print(f"Accuracy on held-out questions: {accuracy(RES0):.1%} -> {accuracy(RES1):.1%}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Iterate to the benchmark

The same loop, automated and started again from the baseline so the whole path is recorded: measure, diagnose, apply the top control, measure again, until held-out accuracy reaches the benchmark or every control has been applied. To check that SHAP is choosing well, each round also tries every remaining control on a scratch copy and records what it would actually recover. On a real project you could not afford that check; here it tells us whether the diagnosis can be trusted.
""")

code(r"""
def wrong_dev(res):
    return res[res.split == "dev"].wrong.mean() * 1000

H, applied, res, diag = G.copy(), [], RES0, D0
HISTORY, CHOICES = [], []
HISTORY.append(dict(round=0, applied="(baseline)", test_accuracy=accuracy(res), dev_accuracy=accuracy(res, "dev"),
                    mean_f1=res[res.split == "test"].f1.mean(), auc=diag["auc"]))
HEALTH_ROUNDS = {"baseline": health(H)}
t0 = time.time()
for rnd in range(1, len(FIXES) + 1):
    if accuracy(res) >= BENCHMARK:
        break
    # What would each remaining control actually recover right now?
    measured = {}
    for c in FIXES:
        if c not in applied:
            scratch = H.copy(); FIXES[c](scratch)
            measured[c] = wrong_dev(res) - wrong_dev(evaluate(scratch))
            del scratch
    choice = next_fix(diag, applied)
    attributed = diag["table"].attributed_wrong_per_1000
    CHOICES.append(dict(round=rnd, shap_choice=choice, best_by_measurement=max(measured, key=measured.get),
                        **{f"attributed: {c}": attributed.get(c, 0.0) for c in measured},
                        **{f"recovered: {c}": round(v, 1) for c, v in measured.items()}))
    result = FIXES[choice](H)
    applied.append(choice)
    res = evaluate(H)
    diag = diagnose(res) if res[res.split == "dev"].wrong.sum() >= 10 else dict(auc=None, table=pd.DataFrame(index=[]))
    HEALTH_ROUNDS[f"after {choice}"] = health(H)
    HISTORY.append(dict(round=rnd, applied=choice, test_accuracy=accuracy(res), dev_accuracy=accuracy(res, "dev"),
                        mean_f1=res[res.split == "test"].f1.mean(), auc=diag["auc"]))
    print(f"Round {rnd}: {choice:22s} {result:32s} held-out accuracy {accuracy(res):.1%}   ({time.time() - t0:.0f}s)")
RES_FINAL, G_FINAL = res, H
HIST = pd.DataFrame(HISTORY)
HIST.round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 3.4))
x = HIST["round"]
ax.plot(x, HIST.test_accuracy * 100, color=BLUE, linewidth=2, marker="o", markersize=7)
ax.axhline(BENCHMARK * 100, color="#8c8b85", linestyle="--", linewidth=1)
ax.text(x.min(), BENCHMARK * 100 + 0.8, f"benchmark {BENCHMARK:.0%}", ha="left", fontsize=9, color="#52514e")
for xi, acc in zip(x, HIST.test_accuracy):
    ax.annotate(f"{acc:.1%}", (xi, acc * 100), textcoords="offset points", xytext=(0, -16), ha="center",
                fontsize=9, color="#0b0b0b")
ax.set_xticks(x, ["baseline" if r == 0 else f"+ {a}" for r, a in zip(x, HIST.applied)], rotation=12, ha="right", fontsize=9)
ax.set_ylabel("Exact-match accuracy, held out (%)")
ax.set_ylim(min(HIST.test_accuracy.min() * 100 - 6, 60), 101)
ax.set_title(f"Held-out accuracy climbs from {HIST.test_accuracy.iloc[0]:.0%} to "
             f"{HIST.test_accuracy.iloc[-1]:.0%} in {len(HIST) - 1} rounds")
plt.tight_layout(); plt.show()

(pd.DataFrame(HEALTH_ROUNDS) * 100).round(2)
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Did the diagnosis pick the right fix?

For each round, the table compares the control SHAP ranked first with the control that would actually have recovered the most wrong answers. Attributed and recovered numbers are not expected to match exactly. Attributed numbers always sum to the total, but recoveries do not, because a question hit by two defects stays wrong until both are fixed. And where one feature responds to several defects at once, as dead ends do to deletions, duplicates and reversed edges alike, attribution spreads blame that a repair separates cleanly. What matters for the loop is whether the ranking points at a control that genuinely helps.

When a round disagrees, the miss is information. Here the diagnosis can under-rank referential integrity: a dropped many-to-many relationship, such as a vendor no longer linked to a product it supplies, leaves no trace on the query's path, because no rule says how many vendors a product must have. The order-subtotal check catches missing order lines, but nothing equivalent exists for supply or bill-of-materials links. The loop still converges, because every round is re-measured against the golden dataset. A team would read the miss as a reason to add a reconciliation check for those relationships.
""")

code(r"""
CH = pd.DataFrame(CHOICES)
CH["agrees"] = CH.shap_choice == CH.best_by_measurement
print(f"SHAP's first choice matched the best measured fix in {CH.agrees.sum()} of {len(CH)} rounds")
CH.set_index("round").T
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. The reveal: what the pipeline actually did

The simulated pipeline injected four kinds of defects. Compare them with what the diagnosis found and the fixes repaired.
""")

code(r"""
FIXED_BY = {"properties dropped": "Completeness", "entities duplicated": "Deduplication",
            "edges reversed": "Schema and hierarchy", "edges dropped": "Referential integrity"}
reveal = pd.DataFrame({"injected": PIPELINE_LOG, "control that fixes it": FIXED_BY})
reveal["round applied"] = reveal["control that fixes it"].map({c: i + 1 for i, c in enumerate(applied)})
display(reveal)
print(f"Held-out accuracy: {HIST.test_accuracy.iloc[0]:.1%} -> {HIST.test_accuracy.iloc[-1]:.1%} "
      f"after {len(applied)} rounds")
left = RES_FINAL[RES_FINAL.wrong == 1]
print(f"\nStill wrong after the loop: {len(left)} of {len(RES_FINAL)} questions")
left.groupby(["template", "kind"]).size().rename("questions").to_frame() if len(left) else None
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. Limits, and using this on your own graph

**What this measures.** Structural quality of a graph, judged by exact answers to structured questions. The question parser is assumed perfect, so these results say nothing about how well an LLM translates questions into graph queries. Test that layer separately, against the same golden dataset.

**The golden dataset defines quality.** Defects that no question touches cost nothing here. A golden set that under-samples a part of the graph will under-report its problems, so cover the questions your users actually ask.

**Synthetic defects, real procedures.** The pipeline's defects are simulated so the diagnosis can be checked against known causes. The fixes are ordinary data-quality procedures, and two of them, completeness and referential integrity, assume a source system to reconcile against.

**Attribution is a ranking, not a forecast.** The attributed numbers say where to look first. The re-measurement after each fix is the number to report.

**To run it on your own graph**, replace four things and keep the rest:

1. `build_graph`: load your entities and relationships into a typed `MultiDiGraph` and record `SCHEMA`.
2. `TEMPLATES` and `GOLD_SQL`: your question paths and an independent source of correct answers.
3. `PROPS` and `CONTROL`: your entity properties and the controls your team owns. Required properties and cardinality rules are learned from the source automatically.
4. `FIXES`: your data-quality procedures. The loop calls them in the order the diagnosis ranks them.

The features, classifier, attribution and loop need no changes, except that the hierarchy features read a level attribute on recursive relationships. Drop them if your graph has no hierarchy.
""")


def build(path=str(Path(__file__).resolve().parent.parent / "kg_diagnosis_adventureworks.ipynb")):
    nb = nbf.v4.new_notebook()
    cells = []
    for k, s, meta in CELLS:
        c = nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
        c.metadata.update(meta)
        cells.append(c)
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": []}
    nbf.write(nb, path)
    print(f"Wrote {path}: {len(CELLS)} cells")


if __name__ == "__main__":
    build()
    if "--exec" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        ns = {"__name__": "__main__"}
        for i, (k, s, _) in enumerate(CELLS):
            if k == "code":
                print(f"--- cell {i}")
                exec(compile(s, f"cell{i}", "exec"), ns)
