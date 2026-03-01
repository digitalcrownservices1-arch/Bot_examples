from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FormulaRenderer:
    def render_integral_image(self, latex_formula: str, width: int = 1400, height: int = 800) -> bytes:
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor("white")
        plt.axis("off")

        plt.text(
            0.5,
            0.5,
            f"${latex_formula}$",
            fontsize=96,
            ha="center",
            va="center",
        )

        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        buffer.seek(0)
        return buffer.read()
