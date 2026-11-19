"""Home screen: new project, open, recent projects."""
from pathlib import Path
from tkinter import ttk

from .. import APP_NAME
from ..core import project as pj


class HomeView(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=40)
        self.app = app
        ttk.Label(self, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(self, text="Review detector output on satellite imagery, fix boxes, flag missed objects, "
                             "and export validated labels.", style="Muted.TLabel").pack(anchor="w", pady=(4, 24))
        row = ttk.Frame(self)
        row.pack(anchor="w", pady=(0, 24))
        ttk.Button(row, text="＋  New project", style="Accent.TButton", command=app.new_project).pack(side="left")
        ttk.Label(row, text="a short guided setup", style="Muted.TLabel").pack(side="left", padx=(6, 18))
        ttk.Button(row, text="Open Project…", command=app.browse_project).pack(side="left", padx=8)

        ttk.Label(self, text="Recent projects", style="H2.TLabel").pack(anchor="w")
        cols = ("name", "images", "path")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=14)
        for c, w, t in (("name", 260, "Project"), ("images", 80, "Images"), ("path", 640, "Folder")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=6)
        self.tree.bind("<Double-1>", self._open_selected)
        self.tree.bind("<Return>", self._open_selected)
        btns = ttk.Frame(self)
        btns.pack(anchor="w")
        ttk.Button(btns, text="Open", command=self._open_selected).pack(side="left")
        ttk.Button(btns, text="Remove from list", command=self._forget).pack(side="left", padx=8)
        self._fill()

    def _fill(self):
        import json
        for p in pj.load_recent():
            meta_path = Path(p) / "project.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except ValueError:
                continue
            self.tree.insert("", "end", iid=p, values=(meta.get("name", Path(p).name),
                                                       len(meta.get("imagery", [])), p))

    def _open_selected(self, _e=None):
        sel = self.tree.selection()
        if sel:
            self.app.open_project(sel[0])

    def _forget(self):
        for s in self.tree.selection():
            pj.remove_recent(s)
            self.tree.delete(s)
