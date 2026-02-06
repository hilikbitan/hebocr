"""Tkinter GUI for subtitle extraction."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from extract import ExtractConfig, extract_subtitles, load_config, save_config
from utils import CancelledError


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Hebrew Subtitle OCR → SRT")
        self.geometry("900x600")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self._build_ui()
        self.after(200, self._drain_log_queue)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.extract_tab = ttk.Frame(notebook)
        self.train_tab = ttk.Frame(notebook)
        self.logs_tab = ttk.Frame(notebook)

        notebook.add(self.extract_tab, text="Extract")
        notebook.add(self.train_tab, text="Train")
        notebook.add(self.logs_tab, text="Logs")

        self._build_extract_tab()
        self._build_train_tab()
        self._build_logs_tab()

    def _build_extract_tab(self) -> None:
        frame = self.extract_tab
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Video File:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.video_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.video_var).grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_video).grid(row=0, column=2, padx=10, pady=10)

        ttk.Label(frame, text="Output SRT:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.output_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.output_var).grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        ttk.Button(frame, text="Browse", command=self._browse_output).grid(row=1, column=2, padx=10, pady=10)

        ttk.Label(frame, text="Config:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.config_var = tk.StringVar(value="config.yaml")
        ttk.Entry(frame, textvariable=self.config_var).grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        ttk.Button(frame, text="Load", command=self._load_config).grid(row=2, column=2, padx=10, pady=10)

        self.manual_roi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Manual ROI", variable=self.manual_roi_var).grid(
            row=3, column=0, padx=10, pady=5, sticky="w"
        )

        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Debug mode", variable=self.debug_var).grid(
            row=3, column=1, padx=10, pady=5, sticky="w"
        )

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(frame, variable=self.progress_var, maximum=1.0)
        self.progress.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        self.start_button = ttk.Button(frame, text="Start Extraction", command=self._start_extraction)
        self.start_button.grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.cancel_button = ttk.Button(frame, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.cancel_button.grid(row=5, column=1, padx=10, pady=10, sticky="w")

    def _build_train_tab(self) -> None:
        ttk.Label(self.train_tab, text="Training is optional. Use train.py for CLI training.").pack(
            padx=20, pady=20, anchor="w"
        )

    def _build_logs_tab(self) -> None:
        self.log_text = tk.Text(self.logs_tab, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi")])
        if path:
            self.video_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".srt", filetypes=[("SRT", "*.srt")])
        if path:
            self.output_var.set(path)

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.config_var.set(path)

    def _start_extraction(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Running", "Extraction already running")
            return
        video = self.video_var.get()
        output = self.output_var.get()
        config_path = self.config_var.get()
        if not video or not output:
            messagebox.showerror("Missing", "Please choose a video and output path")
            return
        config = load_config(config_path) if Path(config_path).exists() else ExtractConfig()
        config.debug = self.debug_var.get()

        self.cancel_event.clear()
        self.progress_var.set(0.0)
        self.start_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        self._log(f"Starting extraction: {video}")

        def worker() -> None:
            try:
                extract_subtitles(
                    video,
                    output,
                    config,
                    progress_cb=self._on_progress,
                    log_cb=self._log,
                    cancel_event=self.cancel_event,
                    manual_roi=self.manual_roi_var.get(),
                )
                self._log("Extraction complete")
            except CancelledError:
                self._log("Extraction cancelled")
            except Exception as exc:  # pylint: disable=broad-except
                self._log(f"Error: {exc}")
            finally:
                self._on_done()

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self._log("Cancel requested")

    def _on_progress(self, value: float, message: str) -> None:
        self.progress_var.set(value)
        self._log(message)

    def _on_done(self) -> None:
        self.start_button.configure(state=tk.NORMAL)
        self.cancel_button.configure(state=tk.DISABLED)

    def _log(self, message: str) -> None:
        self.log_queue.put(message)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.after(200, self._drain_log_queue)


if __name__ == "__main__":
    app = App()
    app.mainloop()
