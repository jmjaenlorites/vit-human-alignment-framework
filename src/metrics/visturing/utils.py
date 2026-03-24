"""Utilidades para resumir resultados agregados de Visturing."""

import re
from typing import Any

import numpy as np
import pandas as pd


def build_evaluation_table(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Construye una tabla resumen con el mismo formato del repo upstream."""

    data = {prop: values["correlations"] for prop, values in data.items()}

    def get_val(val: Any) -> str:
        return f"{val:.2f}" if val is not None and not np.isnan(val) else "-"

    rows = [
        {
            "Property": "Prop. 1",
            "RMSE fit (rho_p)": get_val(data.get("prop1", {}).get("pearson")),
            "Curve Order (rho_k)": "-",
        },
        {
            "Property": "Prop. 2 achrom.",
            "RMSE fit (rho_p)": get_val(data.get("prop2", {}).get("pearson_achrom")),
            "Curve Order (rho_k)": get_val(
                data.get("prop2", {})
                .get("kendall", {})
                .get("achrom", {})
                .get("kendall")
            ),
        },
        {
            "Property": "Prop. 2 chrom.",
            "RMSE fit (rho_p)": get_val(data.get("prop2", {}).get("pearson_chrom")),
            "Curve Order (rho_k)": (
                "RG: "
                f"{get_val(data.get('prop2', {}).get('kendall', {}).get('red_green', {}).get('kendall'))}"
                " | YB: "
                f"{get_val(data.get('prop2', {}).get('kendall', {}).get('yellow_blue', {}).get('kendall'))}"
            ),
        },
        {
            "Property": "Prop. 3 & 4",
            "RMSE fit (rho_p)": get_val(data.get("prop3_4", {}).get("pearson")),
            "Curve Order (rho_k)": get_val(
                data.get("prop3_4", {}).get("kendall", {}).get("kendall")
            ),
        },
        {
            "Property": "Prop. 5",
            "RMSE fit (rho_p)": get_val(data.get("prop5", {}).get("pearson")),
            "Curve Order (rho_k)": get_val(
                data.get("prop5", {}).get("kendall", {}).get("kendall")
            ),
        },
        {
            "Property": "Prop. 6 & 7",
            "RMSE fit (rho_p)": get_val(data.get("prop6_7", {}).get("pearson")),
            "Curve Order (rho_k)": (
                "A: "
                f"{get_val(data.get('prop6_7', {}).get('kendall', {}).get('achrom', {}).get('kendall'))}"
                " | RG: "
                f"{get_val(data.get('prop6_7', {}).get('kendall', {}).get('red_green', {}).get('kendall'))}"
                " | YB: "
                f"{get_val(data.get('prop6_7', {}).get('kendall', {}).get('yellow_blue', {}).get('kendall'))}"
            ),
        },
    ]

    shared_rmse = get_val(data.get("prop8", {}).get("pearson"))
    for prop_key, prop_name in [
        ("prop8", "Prop. 8"),
        ("prop9", "Prop. 9"),
        ("prop10", "Prop. 10"),
    ]:
        rows.append(
            {
                "Property": prop_name,
                "RMSE fit (rho_p)": shared_rmse,
                "Curve Order (rho_k)": (
                    "Low f: "
                    f"{get_val(data.get(prop_key, {}).get('kendall', {}).get('low', {}).get('kendall'))}"
                    " | High f: "
                    f"{get_val(data.get(prop_key, {}).get('kendall', {}).get('high', {}).get('kendall'))}"
                ),
            }
        )

    return pd.DataFrame(rows)


def extract_numbers_from_table(table_results: pd.DataFrame) -> list[float]:
    """Extrae todos los flotantes presentes en la tabla resumen."""

    numbers = []
    for element in table_results.to_numpy().ravel():
        matches = re.findall(r"[-+]?\d+\.\d+", str(element))
        for number in matches:
            numbers.append(float(number))
    return numbers
