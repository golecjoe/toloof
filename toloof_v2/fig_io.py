import json
import matplotlib.pyplot as plt


def save_zernike_table(results, outfile="zernike_table.png", title=None):
    """
    Render a dictionary of fit results as a PNG table using matplotlib.

    Supported input formats
    -----------------------
    1) Row-wise dict:
       {
         "TILT_H": {"ID": 1, "VALUE": 1218.13, "ERROR": 210.37, "SNR": 5.79},
         "TILT_V": {"ID": 2, "VALUE": 1459.16, "ERROR": 110.67, "SNR": 13.18},
         ...
       }

    2) Column-wise dict:
       {
         "LABEL": ["TILT_H", "TILT_V", ...],
         "ID":    [1, 2, ...],
         "VALUE": [1218.13, 1459.16, ...],
         "ERROR": [210.37, 110.67, ...],
         "SNR":   [5.79, 13.18, ...]
       }
    """

    # --------------------------------------------------
    # Normalize input into a list of row dicts
    # --------------------------------------------------
    rows = []

    if isinstance(results, dict):
        vals = list(results.values())

        # Case 1: row-wise nested dict
        if len(vals) > 0 and all(isinstance(v, dict) for v in vals):
            for label, entry in results.items():
                row = {
                    "LABEL": label,
                    "ID": entry.get("ID", entry.get("id", "")),
                    "VALUE": entry.get("VALUE", entry.get("value", "")),
                    "ERROR": entry.get("ERROR", entry.get("error", "")),
                    "SNR": entry.get("SNR", entry.get("snr", "")),
                }
                rows.append(row)

        # Case 2: column-wise dict
        elif all(k in results for k in ["LABEL", "ID", "VALUE", "ERROR", "SNR"]):
            n = len(results["LABEL"])
            for i in range(n):
                row = {
                    "LABEL": results["LABEL"][i],
                    "ID": results["ID"][i],
                    "VALUE": results["VALUE"][i],
                    "ERROR": results["ERROR"][i],
                    "SNR": results["SNR"][i],
                }
                rows.append(row)

        else:
            raise ValueError("Unrecognized dictionary format for results.")

    else:
        raise ValueError("results must be a dictionary.")

    # --------------------------------------------------
    # Format values for display
    # --------------------------------------------------
    def fmt(x, ndig=2):
        if isinstance(x, (int, float)):
            return f"{x:.{ndig}f}" if not isinstance(x, int) else str(x)
        return str(x)

    cell_text = []
    for row in rows:
        cell_text.append([
            str(row["LABEL"]),
            str(row["ID"]),
            fmt(row["VALUE"], 2),
            fmt(row["ERROR"], 2),
            fmt(row["SNR"], 2),
        ])

    col_labels = ["LABEL", "ID", "VALUE", "ERROR", "SNR"]

    # --------------------------------------------------
    # Figure sizing
    # --------------------------------------------------
    nrows = len(cell_text)
    fig_height = max(2.0, 0.38 * (nrows + 1))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    # --------------------------------------------------
    # Build table
    # --------------------------------------------------
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 1.6)

    # --------------------------------------------------
    # Styling to resemble your example
    # --------------------------------------------------
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

        if r == 0:
            cell.set_facecolor("#d9d9d9")  # header
            cell.set_text_props(weight="normal")
        else:
            cell.set_facecolor("#eeeeee")  # body

    if title is not None:
        ax.set_title(title, fontsize=14, pad=12)

    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved table to {outfile}")