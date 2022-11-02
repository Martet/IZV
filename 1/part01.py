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
    """
    Integrates a function using the trapezoidal rule

    :param x: Sorted array of integration points
    :param y: Array of function values on integration points
    :return: Integration result as a float
    """
    return np.sum((x[1:] - x[:-1]) * ((y[:-1] + y[1:]) / 2))


def generate_graph(
    a: List[float] = [1.0, 2.0, -2.0],
    show_figure: bool = False,
    save_path: str | None = None,
):
    """
    Generate a graph of a parametric function f_a(x) = a * x^2

    :param a: List of function parameters as floats
    :param show_figure: Show the graph on True
    :param save_path: Path to save the generated graph
    """
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    inputs = np.tile(np.linspace(-3, 3, 1000), (3, 1))
    outputs = np.array(a).reshape(-1, 1) * np.square(inputs)

    for i in range(len(a)):
        (line,) = ax.plot(
            inputs[i], outputs[i], label=r"$\gamma_{{{}}}(x)$".format(a[i])
        )
        ax.annotate(
            r"$\int f_{{{}}}(x)dx$".format(a[i]),
            xy=(3, line.get_ydata()[-1]),
            xytext=(0, 0),
            textcoords="offset points",
            va="center",
        )
        ax.fill_between(inputs[i], outputs[i], alpha=0.1)

    ax.set_xlim(-3, 4)
    ax.set_ylim(-20, 20)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f_a(x)$")
    ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=3, loc="center")
    ax.xaxis.get_major_ticks()[-1].set_visible(False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    if show_figure:
        fig.show()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Generate a graph of two sinus signals and their sum

    :param show_figure: Show the graph on True
    :param save_path: Path to save the generated graph
    """
    pass


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    """
    Download and parse temperature data from a url

    :param url: Url to download from
    :return: List of dictionaries containing temperature data
    """
    r = requests.get(url)
    r.raise_for_status()
    bs = BeautifulSoup(r.text, "html.parser")
    data = []
    for row in bs.find_all("tr", attrs={"class": "ro1"}):
        cols = [col for col in row.find_all("td") if col.find("p")]
        row_dict = {}
        row_dict["year"] = int(cols[0].p.string)
        row_dict["month"] = int(cols[1].p.string)
        row_dict["temp"] = np.array(
            [float(col.p.string.replace(",", ".")) for col in cols[2:]]
        )
        data.append(row_dict)
    return data


def get_avg_temp(data, year=None, month=None) -> float:
    """
    Get the average temperature with a possibility to filter by year and month

    :param data: List of dicts containing temperature data
    :param year: Filter data by year
    :param month: Filter data by month
    :return: Average temperature as float
    """
    if year is not None:
        data = [d for d in data if d["year"] == year]
    if month is not None:
        data = [d for d in data if d["month"] == month]

    data = np.concatenate([d["temp"] for d in data], axis=None)
    return np.average(data)
