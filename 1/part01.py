#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Martin Zmitko (xzmitk01)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""


from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def integrate(x: np.array, y: np.array) -> float:
    return np.sum((x[1:] - x[:-1]) * ((y[:-1] + y[1:]) / 2))


def generate_graph(
    a: List[float], show_figure: bool = False, save_path: str | None = None
):
    pass


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    pass


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    r = requests.get(url)
    r.raise_for_status()
    bs = BeautifulSoup(r.text, "html.parser")
    data = []
    for row in bs.find_all(attrs={"class": "ro1"}):
        row_dict = {}
        cols = [col for col in row.find_all("td") if col.find("p")]
        row_dict["year"] = int(cols[0].p.string)
        row_dict["month"] = int(cols[1].p.string)
        row_dict["temp"] = np.array(
            [float(col.p.string.replace(",", ".")) for col in cols[2:]]
        )
        data.append(row_dict)
    return data


def get_avg_temp(data, year=None, month=None) -> float:
    if year is not None:
        data = [d for d in data if d["year"] == year]
    if month is not None:
        data = [d for d in data if d["month"] == month]

    data = np.concatenate([d["temp"] for d in data], axis=None)
    return np.average(data)
