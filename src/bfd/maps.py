"""Map of an instance and its optimal network."""


def figure_instance_map(inst, optimum, path="figure2_instance_map.png"):
    """Map of an instance and its optimal network: what is being decided."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    pts, facs = inst["points"], inst["facilities"]
    for i, j in optimum["assignment"].items():
        ax.plot([pts[i, 0], facs[j, 0]], [pts[i, 1], facs[j, 1]],
                color="#999999", linewidth=0.9, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=inst["volumes"] * 1.1, c="#4C72B0",
               alpha=0.85, edgecolor="white", linewidth=0.8, zorder=2,
               label="Collection point (size = return volume)")
    opened = set(optimum["opened"])
    for j in range(inst["n"]):
        is_open = j in opened
        ax.scatter(facs[j, 0], facs[j, 1], s=230, marker="s",
                   c="#C44E52" if is_open else "white",
                   edgecolor="#C44E52", linewidth=1.6, zorder=3)
        ax.annotate(f"F{j}", (facs[j, 0], facs[j, 1]), fontsize=8,
                    ha="center", va="center", zorder=4,
                    color="white" if is_open else "#C44E52")
    ax.scatter([], [], s=120, marker="s", c="#C44E52", label="Facility opened")
    ax.scatter([], [], s=120, marker="s", c="white", edgecolor="#C44E52",
               label="Facility not opened")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title(f"Instance {inst['name']}: optimal network, "
                 f"EUR {optimum['cost']:,.0f} per period", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, columnspacing=1.2)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
