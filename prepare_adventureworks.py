"""Offline data prep for the kg-diagnosis AdventureWorks tutorial.

Run once on the presenter's machine. It downloads Microsoft's AdventureWorks
OLTP install script (MIT licensed), parses the header-less CSV files using the
column lists and delimiters declared in instawdb.sql, keeps the tables and
columns the tutorial graph needs, and writes one Parquet file per table.

Upload the output folder to a Hugging Face dataset repo (or a GitHub release)
and pin its revision in the notebook.

    python prepare_adventureworks.py --out data/aw
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://github.com/Microsoft/sql-server-samples/releases/download/"
    "adventureworks/AdventureWorks-oltp-install-script.zip"
)

# (schema, table) -> columns to keep. Everything else (rowguid, ModifiedDate,
# XML demographics, photos, passwords) is dropped.
KEEP = {
    ("Production", "Product"): [
        "ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag",
        "Color", "StandardCost", "ListPrice", "Size", "Weight",
        "DaysToManufacture", "ProductLine", "Class", "Style",
        "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate",
    ],
    ("Production", "ProductSubcategory"): ["ProductSubcategoryID", "ProductCategoryID", "Name"],
    ("Production", "ProductCategory"): ["ProductCategoryID", "Name"],
    ("Production", "ProductModel"): ["ProductModelID", "Name"],
    ("Production", "BillOfMaterials"): [
        "BillOfMaterialsID", "ProductAssemblyID", "ComponentID", "StartDate",
        "EndDate", "BOMLevel", "PerAssemblyQty",
    ],
    ("Production", "Location"): ["LocationID", "Name", "CostRate", "Availability"],
    ("Production", "ProductInventory"): ["ProductID", "LocationID", "Shelf", "Bin", "Quantity"],
    ("Purchasing", "Vendor"): [
        "BusinessEntityID", "AccountNumber", "Name", "CreditRating",
        "PreferredVendorStatus", "ActiveFlag",
    ],
    ("Purchasing", "ProductVendor"): [
        "ProductID", "BusinessEntityID", "AverageLeadTime", "StandardPrice",
        "MinOrderQty", "MaxOrderQty",
    ],
    ("Purchasing", "PurchaseOrderHeader"): [
        "PurchaseOrderID", "Status", "EmployeeID", "VendorID", "ShipMethodID",
        "OrderDate", "ShipDate", "TotalDue",
    ],
    ("Purchasing", "PurchaseOrderDetail"): [
        "PurchaseOrderID", "PurchaseOrderDetailID", "ProductID", "OrderQty",
        "UnitPrice", "ReceivedQty", "RejectedQty",
    ],
    ("Purchasing", "ShipMethod"): ["ShipMethodID", "Name", "ShipBase", "ShipRate"],
    ("Sales", "Customer"): ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber"],
    ("Sales", "Store"): ["BusinessEntityID", "Name", "SalesPersonID"],
    ("Sales", "SalesPerson"): [
        "BusinessEntityID", "TerritoryID", "SalesQuota", "Bonus",
        "CommissionPct", "SalesYTD",
    ],
    ("Sales", "SalesTerritory"): ["TerritoryID", "Name", "CountryRegionCode", "Group", "SalesYTD"],
    ("Sales", "SalesOrderHeader"): [
        "SalesOrderID", "OrderDate", "DueDate", "ShipDate", "Status",
        "OnlineOrderFlag", "CustomerID", "SalesPersonID", "TerritoryID",
        "ShipMethodID", "SubTotal", "TotalDue",
    ],
    ("Sales", "SalesOrderDetail"): [
        "SalesOrderID", "SalesOrderDetailID", "OrderQty", "ProductID",
        "SpecialOfferID", "UnitPrice", "UnitPriceDiscount",
    ],
    ("Sales", "SpecialOffer"): [
        "SpecialOfferID", "Description", "DiscountPct", "Type", "Category",
        "StartDate", "EndDate",
    ],
    ("Person", "Person"): ["BusinessEntityID", "PersonType", "FirstName", "LastName"],
    ("Person", "CountryRegion"): ["CountryRegionCode", "Name"],
    ("HumanResources", "Employee"): [
        "BusinessEntityID", "JobTitle", "HireDate", "SalariedFlag",
        "OrganizationNode", "CurrentFlag",
    ],
    ("HumanResources", "Department"): ["DepartmentID", "Name", "GroupName"],
    ("HumanResources", "EmployeeDepartmentHistory"): [
        "BusinessEntityID", "DepartmentID", "ShiftID", "StartDate", "EndDate",
    ],
}


def fetch_archive(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / "AdventureWorks-oltp-install-script.zip"
    if not zpath.exists():
        print(f"Downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, zpath)
    src = cache / "src"
    if not (src / "instawdb.sql").exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(src)
    return src


def table_columns(sql: str, schema: str, table: str, computed: bool = False) -> list[str]:
    """Declared columns in order; computed (AS ...) columns only if asked.

    Some CSVs in the install script carry computed columns and some do not,
    so read_table tries both lists against the field count.
    """
    m = re.search(
        rf"CREATE TABLE \[{schema}\]\.\[{table}\]\s*\((.*?)\n\)\s*ON", sql, re.S
    )
    if not m:
        raise ValueError(f"No CREATE TABLE for {schema}.{table}")
    cols = []
    for line in m.group(1).splitlines():
        line = line.strip()
        cm = re.match(r"\[(\w+)\]\s+(.*)", line)
        if not cm:
            continue
        name, rest = cm.groups()
        if rest.upper().startswith("AS ") and not computed:
            continue
        cols.append(name)
    return cols


def bulk_spec(sql: str, schema: str, table: str) -> tuple[str, str, str]:
    m = re.search(
        rf"BULK INSERT \[{schema}\]\.\[{table}\] FROM '\$\(SqlSamplesSourceDataPath\)(\w+\.csv)'"
        rf"\s*WITH \((.*?)\)",
        sql,
        re.S,
    )
    if not m:
        raise ValueError(f"No BULK INSERT for {schema}.{table}")
    fname, opts = m.groups()
    ft = re.search(r"FIELDTERMINATOR\s*=\s*'([^']*)'", opts).group(1)
    rt = re.search(r"ROWTERMINATOR\s*=\s*'([^']*)'", opts).group(1)
    widechar = "widechar" in opts
    ft = ft.replace("\\t", "\t")
    rt = {"0x0a": "\n", "\\n": "\n"}.get(rt, rt.replace("\\n", "\n"))
    return fname, ft, rt + ("|wide" if widechar else "")


def read_table(src: Path, sql: str, schema: str, table: str) -> pd.DataFrame:
    stored = table_columns(sql, schema, table)
    every = table_columns(sql, schema, table, computed=True)
    fname, ft, rt = bulk_spec(sql, schema, table)
    wide = rt.endswith("|wide")
    rt = rt.replace("|wide", "")
    raw = (src / fname).read_bytes()
    text = raw.decode("utf-16" if wide else "utf-8-sig", errors="replace")
    rows = [r for r in text.split(rt) if r.strip()]
    n0 = len(rows[0].rstrip("\r").split(ft))
    cols = every if n0 == len(every) else stored
    records = []
    for r in rows:
        fields = r.rstrip("\r").split(ft)
        if len(fields) != len(cols):
            raise ValueError(
                f"{schema}.{table}: expected {len(cols)} fields, got {len(fields)}: {r[:120]!r}"
            )
        records.append(fields)
    df = pd.DataFrame(records, columns=cols)
    df = df.replace("", pd.NA)
    for c in df.columns:  # numeric where every non-null value parses
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() == df[c].notna().sum():
            df[c] = conv
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/aw")
    ap.add_argument("--cache", default=".aw_cache")
    args = ap.parse_args()

    src = fetch_archive(Path(args.cache))
    sql = (src / "instawdb.sql").read_text(encoding="utf-8", errors="replace")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for (schema, table), keep in KEEP.items():
        df = read_table(src, sql, schema, table)[keep]
        df.to_parquet(out / f"{table}.parquet", index=False)
        print(f"{schema}.{table:28s} {len(df):>7,} rows  {len(keep)} cols")

    total = sum(p.stat().st_size for p in out.glob("*.parquet"))
    print(f"\nWrote {len(KEEP)} tables, {total / 1e6:.1f} MB to {out}")


if __name__ == "__main__":
    main()
