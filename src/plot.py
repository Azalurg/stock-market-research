import matplotlib.pyplot as plt

# Global dark mode for all matplotlib plots in this notebook
plt.style.use("dark_background")
plt.rcParams.update(
    {
        "figure.facecolor": "#121212",
        "axes.facecolor": "#121212",
        "axes.edgecolor": "#bbbbbb",
        "axes.labelcolor": "#f0f0f0",
        "xtick.color": "#d9d9d9",
        "ytick.color": "#d9d9d9",
        "grid.color": "#666666",
        "text.color": "#f0f0f0",
    }
)

# Plot function
def plot(
    plot_values: list[float],
    plot_days: list[str],
    title: str,
    stats: dict[str, float] | None = None,
    save: bool = False,
) -> None:
    """Plot portfolio values over time with optional statistics."""
    BG_COLOR = "#121212"
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.plot(plot_days, plot_values, label="Portfolio Value", color="aqua", linewidth=2)
    ax.set_title(title, fontsize=20, color="white")
    if stats:
        stats_lines = [
            f"CAGR: {stats['cagr [%]']}%",
            f"Sharpe: {stats['sharpe_ratio']}",
            f"Max DD: {stats['max_drawdown [%]']}%",
            f"Win Rate: {stats['win_rate']}%",
        ]
        if "transaction_costs_paid" in stats:
            stats_lines.append(f"Costs: {stats['transaction_costs_paid']}")
        if "trades_executed" in stats:
            stats_lines.append(f"Trades: {stats['trades_executed']}")
        stats_text = "\n".join(stats_lines)
        ax.text(
            0.02,
            0.9,
            stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#1e1e1e",
                edgecolor="#666666",
                alpha=0.9,
            ),
        )
    ax.set_xlabel("Time", fontsize=14, color="white")
    ax.set_ylabel("Value", fontsize=14, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444444")
    # Adjust xticks to avoid overlap
    n_ticks = 10
    step = max(1, len(plot_days) // n_ticks)
    ax.set_xticks(plot_days[::step])
    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        loc="upper left", facecolor="#1e1e1e", edgecolor="#666666", labelcolor="white"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#666666")
    plt.tight_layout()
    # Save with unique filename derived from title
    if save:
        filename = (
            title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        )
        plt.savefig(filename, dpi=600, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()