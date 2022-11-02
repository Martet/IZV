#!/usr/bin/env python3
"""
IZV project, part01
Author: Martin Zmitko (xzmitk01)
"""


from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
    t = np.linspace(0, 100, 5000)
    f1 = 0.5 * np.sin(1 / 50 * np.pi * t)
    f2 = 0.25 * np.sin(np.pi * t)
    f3 = f1 + f2
    f3_greater = np.ma.masked_less(f3, f1 - 0.01)
    f3_less = np.ma.masked_greater(f3, f1 + 0.01)

    fig, axes = plt.subplots(nrows=3, figsize=(6, 8))
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel(r"$t$")
        ax.yaxis.set_major_locator(MultipleLocator(0.4))
    axes[0].plot(t, f1)
    axes[0].set_ylabel(r"$f_1(t)$")
    axes[1].plot(t, f2)
    axes[1].set_ylabel(r"$f_2(t)$")
    axes[2].plot(t, f3_less, color="red")
    axes[2].plot(t, f3_greater, color="green")
    axes[2].set_ylabel(r"$f_1(t) + f_2(t)$")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    if show_figure:
        fig.show()


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
