"""Stats tab: precision/recall per source and area, types, objects over time."""
from datetime import datetime
from tkinter import ttk

from ..core import project as pj

# categorical palette in fixed slot order (validated light-mode reference palette; slot 3 aqua is
# < 3:1 on white, so every line is also direct-labelled and the table repeats the numbers).
# An area keeps its colour while the project's set of areas is unchanged (slot = sorted position).
AREA_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def site_color(site, all_sites):
    """Fixed colour per area (slot = position in the sorted area list; 9+ areas reuse slots)."""
    order = sorted(set(all_sites))
    return AREA_COLORS[order.index(site) % len(AREA_COLORS)] if site in order else "#888"


class StatsPanel(ttk.Frame):
    """Live stats: detector precision by source/port, recall against missed objects you added,
    object classes, and objects per image over time."""

    def __init__(self, master, project):
        super().__init__(master, padding=(0, 8, 0, 0))
        self.project = project
        self.canvas_widget = None
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Label(top, text="Precision = confirmed / (confirmed + rejected). Recall = detector hits / (hits + "
                            "missed objects you added), only over images whose scene check is done.",
                  style="Muted.TLabel", wraplength=1000).pack(side="left", padx=10)
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(6, 0))
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self.right = ttk.Frame(body)
        self.right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        def table(title, cols, height):
            ttk.Label(left, text=title, style="H2.TLabel").pack(anchor="w", pady=(6, 2))
            t = ttk.Treeview(left, columns=[c for c, _, _ in cols], show="headings", height=height)
            for c, w, h in cols:
                t.heading(c, text=h)
                t.column(c, width=w, anchor="w" if c in ("site", "source", "cls") else "center")
            t.pack(anchor="w")
            return t
        self.t_src = table("Detector precision by area and source",
                           (("site", 150, "Area"), ("source", 90, "Source"), ("valid", 82, "Confirmed"),
                            ("invalid", 72, "Rejected"), ("undecided", 82, "Undecided"),
                            ("precision", 82, "Precision")), 7)
        self.t_site = table("By area", (("site", 150, "Area"), ("checked", 110, "Scene-checked"),
                                        ("hits", 60, "Hits"), ("added", 65, "Missed"), ("recall", 60, "Recall"),
                                        ("objects", 65, "Objects")), 4)
        self.t_cls = table("Object types", (("cls", 150, "Type"), ("n", 70, "Objects")), 6)

    @staticmethod
    def _pct(v):
        return "-" if v is None else f"{v:.0%}"

    def refresh(self):
        st = pj.compute_stats(self.project)
        for t in (self.t_src, self.t_site, self.t_cls):
            t.delete(*t.get_children())
        for r in st["by_source"]:
            self.t_src.insert("", "end", values=(r["site"], r["source"], r["valid"], r["invalid"], r["undecided"],
                                                 self._pct(r["precision"])))
        for r in st["by_site"]:
            self.t_site.insert("", "end", values=(r["site"], f"{r['checked']}/{r['images']}", r["hits"], r["added"],
                                                  self._pct(r["recall"]), r["objects"]))
        for c, n in sorted(st["classes"].items(), key=lambda kv: -kv[1]):
            self.t_cls.insert("", "end", values=(c, n))
        self._chart(st["series"])

    def _chart(self, series):
        """Objects per image over time, one line per port. Filled marker = complete image,
        hollow = not fully reviewed yet (its count can still go up)."""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.dates as mdates
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator

        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()
        fig = Figure(figsize=(6.4, 4.6), dpi=100, facecolor="#fcfcfb")
        ax = fig.add_subplot(111, facecolor="#fcfcfb")
        ink, muted = "#1a1a19", "#6b6a63"
        points = []   # (site, xs, ys, complete) for hover
        ends = []     # (y, site, x) for direct labels
        all_sites = list(series)
        for site in sorted(series):
            # only images with review work: an untouched image would plot as a false 0
            rows = [r for r in series[site] if r[3]]
            if not rows:
                continue
            xs = [datetime.fromisoformat(d) for d, _, _, _ in rows]
            ys = [v for _, v, _, _ in rows]
            col = site_color(site, all_sites)
            ax.plot(xs, ys, color=col, linewidth=2, zorder=2, label=site)
            done = [(x, y) for (x, y, (_, _, c, _)) in zip(xs, ys, rows) if c]
            todo = [(x, y) for (x, y, (_, _, c, _)) in zip(xs, ys, rows) if not c]
            if done:
                ax.scatter(*zip(*done), s=42, color=col, edgecolors="#fcfcfb", linewidths=2, zorder=3)
            if todo:
                ax.scatter(*zip(*todo), s=42, facecolors="#fcfcfb", edgecolors=col, linewidths=2, zorder=3)
            ends.append((ys[-1], site, xs[-1]))
            points.append((site, xs, ys, [c for _, _, c, _ in rows]))
        ax.set_title("Objects per image over time (reviewed images only)", loc="left", fontsize=11, color=ink,
                     pad=22)
        ax.set_ylabel("objects in image", fontsize=9, color=muted)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)
        loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c0")
        ax.tick_params(colors=muted, labelsize=8)
        if len(points) > 1:   # one series: the direct label names it, no legend box
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8, frameon=False,
                      labelcolor=ink)
        if points:
            # direct labels at each line's last point, nudged apart so they never overlap
            ends.sort()
            last = None
            for y, site, x in ends:
                py = ax.transData.transform((0, y))[1]
                if last is not None and py - last < 12:
                    py = last + 12
                last = py
                dy = py - ax.transData.transform((0, y))[1]
                ax.annotate(site.replace("_", " "), (x, y), xytext=(8, dy), textcoords="offset points",
                            va="center", fontsize=8, color=ink)
        else:
            ax.text(0.5, 0.5, "No reviewed images yet", transform=ax.transAxes, ha="center", color=muted)
        fig.text(0.01, 0.01, "filled = image complete   hollow = not fully reviewed yet (count may rise)",
                 fontsize=8, color=muted)
        fig.tight_layout(rect=(0, 0.04, 0.84, 1))   # right margin for direct labels
        tip = ax.annotate("", (0, 0), xytext=(10, 10), textcoords="offset points", fontsize=8, color=ink,
                          bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#c9c8c0"), zorder=5)
        tip.set_visible(False)
        canvas = FigureCanvasTkAgg(fig, master=self.right)

        def on_move(event):
            show = False
            if event.inaxes is ax:
                best = None
                for site, xs, ys, comp in points:
                    for x, y, c in zip(xs, ys, comp):
                        px, py = ax.transData.transform((mdates.date2num(x), y))
                        d = (px - event.x) ** 2 + (py - event.y) ** 2
                        if d < 144 and (best is None or d < best[0]):   # within 12 px
                            best = (d, site, x, y, c)
                if best:
                    _, site, x, y, c = best
                    tip.xy = (x, y)
                    tip.set_text(f"{site}\n{x:%Y-%m-%d}: {y} object{'s' if y != 1 else ''}\n"
                                 f"{'complete' if c else 'not fully reviewed'}")
                    show = True
            if show != tip.get_visible() or show:
                tip.set_visible(show)
                canvas.draw_idle()
        canvas.mpl_connect("motion_notify_event", on_move)
        canvas.mpl_connect("resize_event", lambda e: (fig.tight_layout(rect=(0, 0.04, 0.84, 1)), canvas.draw_idle()))
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_widget = canvas
