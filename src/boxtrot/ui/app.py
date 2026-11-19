"""Main window: screen switching, menus, project open/close."""
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .. import APP_NAME, __version__, config
from ..core import project as pj
from .common import SHORTCUTS, open_folder
from .dashboard import ProjectView
from .dialogs import AddImageryDialog, ExportDialog, SettingsDialog
from .home import HomeView
from .review import ReviewView
from .wizard import SetupWizard


class App:
    def __init__(self, root, project_dir=None):
        self.root = root
        self.project = None
        self.view = None
        self.assist_job = None   # running assist.py train/predict subprocess (survives screen changes)
        root.title(APP_NAME)
        root.geometry("1280x900")
        root.minsize(980, 640)
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("H2.TLabel", font=("TkDefaultFont", 12, "bold"))
        style.configure("Muted.TLabel", foreground="#666")
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))
        style.configure("Treeview", rowheight=22)
        self._build_menu()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        if project_dir:
            self.open_project(project_dir)
        else:
            self.show_home()

    def _build_menu(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="New Project (guided setup)…", command=self.new_project)
        f.add_command(label="Open Project…", command=self.browse_project)
        f.add_command(label="Close Project", command=self.close_project)
        f.add_separator()
        f.add_command(label="Quit", command=self.quit)
        m.add_cascade(label="File", menu=f)
        p = tk.Menu(m, tearoff=0)
        p.add_command(label="Add Images…", command=lambda: self._with_project(self.add_imagery))
        p.add_command(label="Import Review Data…", command=lambda: self._with_project(self.import_review_data))
        p.add_command(label="Settings…", command=lambda: self._with_project(self.edit_settings))
        p.add_command(label="Export Labels…", command=lambda: self._with_project(self.export_labels))
        p.add_command(label="Open Project Folder", command=lambda: self._with_project(
            lambda: open_folder(self.project.root)))
        m.add_cascade(label="Project", menu=p)
        r = tk.Menu(m, tearoff=0)
        r.add_command(label="Resume Review", command=lambda: self._with_project(self.start_review))
        r.add_command(label="Review Added Boxes", command=lambda: self._with_project(
            lambda: self.start_review(mode="additions")))
        m.add_cascade(label="Review", menu=r)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Keyboard Shortcuts", command=lambda: messagebox.showinfo("Keyboard shortcuts", SHORTCUTS))
        h.add_command(label=f"About {APP_NAME}", command=self.about)
        m.add_cascade(label="Help", menu=h)
        self.root.config(menu=m)

    def about(self):
        from .. import __version__
        from ..core import backends
        found = list(backends.available())
        messagebox.showinfo(f"About {APP_NAME}", f"{APP_NAME} {__version__}\n\nReview detector output on georeferenced "
                            f"imagery and build labelled datasets.\nMIT License.\n\nTraining add-on: "
                            + (", ".join(found) if found else "not installed (optional)"))

    def _with_project(self, fn):
        if self.project is None:
            messagebox.showinfo("No project", "Open or create a project first.")
            return
        fn()

    def _swap(self, view):
        if self.view is not None:
            if hasattr(self.view, "deactivate"):
                self.view.deactivate()
            self.view.destroy()
        self.view = view
        view.pack(fill="both", expand=True)

    # -- screens --
    def show_home(self):
        self.project = None
        self.root.title(APP_NAME)
        self._swap(HomeView(self.root, self))

    def show_project(self, tab=None, welcome=None):
        self.root.title(f"{APP_NAME} — {self.project.name}")
        self._swap(ProjectView(self.root, self, tab=tab, welcome=welcome))

    def start_review(self, image_key=None, mode="detections", order=None):
        queue = pj.build_queue(self.project)
        if not queue:
            messagebox.showinfo("Nothing to review", "This project has no imagery yet (or scene checks are off "
                                                     "and no image has detections). Use Add Imagery first.")
            return
        if mode == "additions" and not self.project.additions["features"]:
            messagebox.showinfo("No added boxes", "No manually-added boxes to review yet.")
            return
        self._swap(ReviewView(self.root, self, queue, image_key=image_key, mode=mode, order=order))

    # -- project actions --
    def new_project(self):
        dlg = SetupWizard(self.root)
        self.root.wait_window(dlg)
        if dlg.result:
            self.project = dlg.result
            self.show_project(tab="Model" if dlg.start_mode == "assist" else None, welcome=dlg.start_mode)

    def browse_project(self):
        d = filedialog.askdirectory(title="Open project folder (contains project.json)",
                                    initialdir=str(config.default_projects_dir()
                                                   if config.default_projects_dir().exists() else Path.home()))
        if d:
            self.open_project(d)

    def open_project(self, path):
        if not (Path(path) / "project.json").exists():
            messagebox.showerror("Not a project", f"No project.json in {path}")
            pj.remove_recent(path)
            if self.project is None:
                self.show_home()
            return
        try:
            self.project = pj.Project(path)
        except Exception as exc:
            messagebox.showerror("Could not open project", f"{exc.__class__.__name__}: {exc}")
            return
        pj.add_recent(self.project.root)
        self.show_project()

    def close_project(self):
        self.show_home()

    def add_imagery(self):
        dlg = AddImageryDialog(self.root, self.project)
        self.root.wait_window(dlg)
        if dlg.added:
            self.show_project()

    def import_review_data(self):
        d = filedialog.askdirectory(title="Folder containing review_state.json / manual_additions.geojson",
                                    initialdir=str(self.project.root.parent))
        if not d:
            return
        if not (Path(d) / "review_state.json").exists() and not (Path(d) / "manual_additions.geojson").exists():
            messagebox.showerror("Nothing to import", "That folder has no review_state.json or manual_additions.geojson.")
            return
        n_state, n_add = self.project.import_review_data(d)
        messagebox.showinfo("Imported", f"Imported {n_state} review records and {n_add} added boxes.\n\n"
                                        "Records only show up for detections queued in this project "
                                        "(ids are location-based).")
        self.show_project()

    def edit_settings(self):
        dlg = SettingsDialog(self.root, self.project)
        self.root.wait_window(dlg)
        if dlg.changed:
            self.show_project()

    def export_labels(self):
        dlg = ExportDialog(self.root, self.project)
        self.root.wait_window(dlg)

    def quit(self):
        if self.assist_job and self.assist_job["proc"].poll() is None and not messagebox.askyesno(
                "Job running", f"A {self.assist_job['cmd']} job is still running. It keeps running in the "
                               f"background after you quit, and the Model tab picks up its result next time.\n\nQuit?"):
            return
        if self.view is not None and hasattr(self.view, "deactivate"):
            self.view.deactivate()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(prog="boxtrot", description=f"{APP_NAME}: review detector output on "
                                 "georeferenced imagery and build labelled datasets.")
    ap.add_argument("project", nargs="?", help="project folder to open directly (contains project.json)")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args.project)
    root.mainloop()
