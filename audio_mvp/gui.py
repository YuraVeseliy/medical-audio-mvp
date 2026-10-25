"""Оконный сценарий. Все обращения к Tk выполняются в основном потоке.

Длительные операции передают результат через очередь; обработка моделей и запись
не запускаются при создании окна. Микрофон включается только кнопкой пользователя.
"""
from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .model import AppError, Document, SECTIONS
from .providers import FasterWhisperRecognizer, LocalRuntime
from .providers_indexed import IndexedExtractor
from .recording import AudioRecorder
from .session import Session
from .storage import export_text


class AudioApp:
    """GUI над публичными операциями Session; зависимости можно передать явно."""

    def __init__(self, root: tk.Tk, project_root: str | Path | None = None,
                 session: Session | None = None, runtime: LocalRuntime | None = None,
                 recorder: AudioRecorder | None = None,
                 runtime_root: str | Path | None = None, port: int = 8081,
                 asr_model: str | Path | None = None, device: str = "cpu",
                 compute_type: str = "int8") -> None:
        self.root = root
        default_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
        self.project_root = Path(project_root or default_root)
        runtime_path = Path(runtime_root or os.environ.get("MEDICAL_AUDIO_RUNTIME") or self.project_root / ".runtime")
        self.runtime = runtime or LocalRuntime(runtime_path, port=port)
        self.session = session or Session(
            FasterWhisperRecognizer(asr_model or runtime_path / "models" / "faster-whisper-small",
                                    device=device, compute_type=compute_type),
            IndexedExtractor(self.runtime.base_url),
        )
        self.recorder = recorder or AudioRecorder()
        self.audio_path: Path | None = None
        self.session_path: Path | None = None
        self._busy = False
        self._closing = False
        self._close_when_done = False
        self._loading = False
        self._rendered_document: Document | None = None
        self._events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._device_map: dict[str, int | None] = {"Системный микрофон": None}
        self._record_started = 0.0
        self._evidence_rows: dict[str, Any] = {}
        self.status = tk.StringVar(value="Выберите аудиофайл или начните запись.")
        self.source = tk.StringVar(value="Аудиофайл не выбран")
        self.device_name = tk.StringVar(value="Системный микрофон")
        self.notice = tk.StringVar(value="Выделенные сведения — черновик. Проверьте их по исходной записи.")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-s>", lambda event: self.save_session())
        self.root.bind("<Control-o>", lambda event: self.open_session())
        self.root.bind("<Control-n>", lambda event: self.new_session())
        self._apply_state()
        self.root.after(80, self._drain_events)

    def _build(self) -> None:
        self.root.title("Медицинский аудиомодуль · учебный прототип")
        self.root.geometry("1180x820")
        self.root.minsize(940, 650)
        self.root.configure(background="white")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white", foreground="#17343c", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"), foreground="#087e8b")
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", foreground="#516b73", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("Accent.TButton", background="#087e8b", foreground="white")
        style.map("Accent.TButton", background=[("active", "#06616c"), ("disabled", "#dce8ea")],
                  foreground=[("disabled", "#687e84")])
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=27,
                        background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

        outer = ttk.Frame(self.root, padding=(22, 16))
        outer.grid(sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)
        ttk.Label(outer, text="Диалог → текст → разделы формы", style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(outer, text="Локальная обработка • Жалобы и анамнез • Проверка человеком",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 12))

        files = ttk.Frame(outer)
        files.grid(row=2, column=0, sticky="ew")
        self.new_button = ttk.Button(files, text="Новый сеанс", command=self.new_session)
        self.new_button.pack(side="left", padx=(0, 6))
        self.open_button = ttk.Button(files, text="Открыть сеанс…", command=self.open_session)
        self.open_button.pack(side="left", padx=(0, 6))
        self.save_button = ttk.Button(files, text="Сохранить сеанс…", command=lambda: self.save_session(True))
        self.save_button.pack(side="left", padx=(0, 6))
        self.export_button = ttk.Button(files, text="Экспорт TXT…", command=self.export_current)
        self.export_button.pack(side="left")

        audio = ttk.Frame(outer, padding=(0, 12, 0, 8))
        audio.grid(row=3, column=0, sticky="ew")
        audio.columnconfigure(1, weight=1)
        ttk.Label(audio, text="Микрофон:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.device_combo = ttk.Combobox(audio, textvariable=self.device_name, state="readonly",
                                         values=list(self._device_map))
        self.device_combo.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.refresh_button = ttk.Button(audio, text="Обновить устройства", command=self.refresh_devices)
        self.refresh_button.grid(row=0, column=2, padx=(0, 6))
        self.record_button = ttk.Button(audio, text="Начать запись", command=self.start_recording)
        self.record_button.grid(row=0, column=3, padx=(0, 6))
        self.stop_button = ttk.Button(audio, text="Остановить и сохранить…", command=self.stop_recording)
        self.stop_button.grid(row=0, column=4)
        self.discard_button = ttk.Button(audio, text="Отменить запись", command=self.discard_recording)
        self.discard_button.grid(row=1, column=4, sticky="e", pady=(6, 0))
        self.import_button = ttk.Button(audio, text="Загрузить аудиофайл…", command=self.import_audio)
        self.import_button.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.process_button = ttk.Button(audio, text="Обработать аудио", style="Accent.TButton",
                                          command=self.process_audio)
        self.process_button.grid(row=1, column=2, columnspan=2, sticky="e", padx=(0, 6), pady=(6, 0))
        self.source_label = ttk.Label(outer, textvariable=self.source, style="Muted.TLabel", wraplength=1080)
        self.source_label.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        panels = ttk.Panedwindow(outer, orient="horizontal")
        panels.grid(row=5, column=0, sticky="nsew")
        left = ttk.Frame(panels, padding=(0, 0, 12, 0))
        right = ttk.Frame(panels, padding=(12, 0, 0, 0))
        panels.add(left, weight=1)
        panels.add(right, weight=1)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=2)
        right.rowconfigure(3, weight=2)
        right.rowconfigure(5, weight=1)
        ttk.Label(left, text="Исходная транскрипция", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6))
        self.transcript = self._text(left, row=1, height=16)
        self.transcript.tag_configure("evidence", background="#b8edf0", foreground="#12373f")
        self.transcript.configure(state="disabled")
        ttk.Label(left, text="Нажмите на цитату справа, чтобы увидеть её в исходном тексте.",
                  style="Muted.TLabel", wraplength=450).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.editors: dict[str, tk.Text] = {}
        for row, (section, title) in zip((0, 2), SECTIONS.items()):
            ttk.Label(right, text=title + " — редактируемый текст", style="Section.TLabel").grid(
                row=row, column=0, sticky="w", pady=(0 if row == 0 else 10, 6))
            widget = self._text(right, row=row + 1, height=5)
            widget.bind("<<Modified>>", self._editor_changed)
            self.editors[section] = widget
        ttk.Label(right, text="Цитаты из исходного текста", style="Section.TLabel").grid(
            row=4, column=0, sticky="w", pady=(10, 6))
        evidence_frame = ttk.Frame(right)
        evidence_frame.grid(row=5, column=0, sticky="nsew")
        evidence_frame.columnconfigure(0, weight=1)
        evidence_frame.rowconfigure(0, weight=1)
        self.evidence = ttk.Treeview(evidence_frame, columns=("section", "quote"), show="headings", height=4)
        self.evidence.heading("section", text="Раздел")
        self.evidence.heading("quote", text="Цитата — нажмите для перехода")
        self.evidence.column("section", width=88, minwidth=65, stretch=False)
        self.evidence.column("quote", width=330, minwidth=120)
        self.evidence.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(evidence_frame, orient="vertical", command=self.evidence.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.evidence.configure(yscrollcommand=scrollbar.set)
        self.evidence.bind("<<TreeviewSelect>>", self.show_evidence)
        self.notice_label = ttk.Label(outer, textvariable=self.notice, style="Muted.TLabel", wraplength=1080)
        self.notice_label.grid(row=6, column=0, sticky="ew", pady=(10, 6))
        bottom = ttk.Frame(outer)
        bottom.grid(row=7, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(bottom, textvariable=self.status, wraplength=880)
        self.status_label.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=140)
        self.progress.grid(row=0, column=1, padx=(12, 0))
        outer.bind("<Configure>", self._resize_labels)

    def _resize_labels(self, event: tk.Event) -> None:
        width = max(300, event.width - 44)
        self.source_label.configure(wraplength=width)
        self.notice_label.configure(wraplength=width)
        self.status_label.configure(wraplength=max(250, width - 165))

    def _text(self, parent: ttk.Frame, row: int, height: int) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        widget = tk.Text(frame, wrap="word", font=("Segoe UI", 11), height=height, width=36,
                         background="white", foreground="#17343c", insertbackground="#087e8b",
                         borderwidth=1, relief="solid", padx=10, pady=8, undo=True,
                         selectbackground="#b8edf0")
        widget.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        widget.configure(yscrollcommand=scrollbar.set)
        return widget

    def _record_pending(self) -> bool:
        return self.recorder.status in {"recording", "captured"}

    def _idle(self) -> bool:
        if self._busy or self._closing:
            return False
        if self._record_pending():
            self.status.set("Сначала сохраните или явно отмените текущую запись.")
            return False
        return True

    def _apply_state(self) -> None:
        recording = self._record_pending()
        ready = not self._busy and not recording and not self._closing
        for button in (self.new_button, self.open_button, self.import_button,
                       self.record_button, self.refresh_button):
            button.configure(state="normal" if ready else "disabled")
        self.device_combo.configure(state="readonly" if ready else "disabled")
        self.stop_button.configure(state="normal" if recording and not self._busy else "disabled",
                                   text="Сохранить записанный звук…" if self.recorder.status == "captured"
                                   else "Остановить и сохранить…")
        self.discard_button.configure(state="normal" if recording and not self._busy else "disabled")
        self.process_button.configure(state="normal" if ready and self.audio_path is not None else "disabled")
        has_result = self.session.document is not None
        for button in (self.save_button, self.export_button):
            button.configure(state="normal" if ready and has_result else "disabled")
        for editor in self.editors.values():
            editor.configure(state="normal" if ready and has_result else "disabled")
        if self._busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        self._update_title()

    def _update_title(self) -> None:
        changed = self.session.document is not None and (self.session.dirty or self.session_path is None)
        self.root.title(("* " if changed else "") + "Медицинский аудиомодуль · учебный прототип")

    def _submit(self, text: str, operation: Callable[[], Any],
                success: Callable[[Any], None]) -> None:
        if self._busy:
            return
        self._busy = True
        self.status.set(text)
        self._apply_state()

        def worker() -> None:
            try:
                result = operation()
            except Exception as exc:
                logging.exception("Операция приложения завершилась ошибкой")
                self._events.put(("error", exc))
            else:
                self._events.put(("done", success, result))

        threading.Thread(target=worker, name="audio-processing", daemon=False).start()

    def _drain_events(self) -> None:
        if self._closing:
            return
        try:
            while True:
                event = self._events.get_nowait()
                if event[0] == "status":
                    self.status.set(event[1])
                    continue
                self._busy = False
                if event[0] == "done":
                    try:
                        event[1](event[2])
                    except Exception as exc:
                        logging.exception("Ошибка отображения результата")
                        self._error(exc)
                else:
                    self._error(event[1])
                self._apply_state()
                if self._close_when_done:
                    self._close_when_done = False
                    self.close()
                    if self._closing:
                        return
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _error(self, exc: Exception) -> None:
        if self.session.document is not None and self.session.document is not self._rendered_document:
            self.session_path = None
            self._render_document(self.session.document)
        text = str(exc) if isinstance(exc, AppError) else (
            f"Операция не завершена ({type(exc).__name__}). Данные не удалены; можно повторить попытку.")
        self.status.set(text)
        messagebox.showerror("Не удалось выполнить действие", text, parent=self.root)

    def _sync_edits(self) -> None:
        if self.session.document is not None:
            for key, editor in self.editors.items():
                self.session.edit(key, editor.get("1.0", "end-1c"))

    def _editor_changed(self, event: tk.Event) -> None:
        widget = event.widget
        changed = widget.edit_modified()
        widget.edit_modified(False)
        if not changed or self._loading or self._busy or self.session.document is None:
            return
        self._sync_edits()
        self._update_title()

    def _confirm_unsaved(self) -> bool:
        self._sync_edits()
        if self.session.document is None or (not self.session.dirty and self.session_path is not None):
            return True
        answer = messagebox.askyesnocancel("Результат не сохранён",
            "Сохранить сеанс с текущими правками перед продолжением?\n"
            "«Нет» — продолжить без сохранения, «Отмена» — вернуться к результату.", parent=self.root)
        if answer is None:
            return False
        if answer:
            return self.save_session()
        # Session запрещает перезаписывать dirty-документ. «Нет» выше является
        # явным согласием пользователя отказаться от текущего несохранённого результата.
        self.session.new(force=True)
        self.session_path = None
        self._render_document(None)
        return True

    def _render_document(self, document: Document | None) -> None:
        self._rendered_document = document
        self._loading = True
        try:
            self.transcript.configure(state="normal")
            self.transcript.delete("1.0", "end")
            self.transcript.insert("1.0", document.transcript if document else "")
            self.transcript.tag_remove("evidence", "1.0", "end")
            self.transcript.configure(state="disabled")
            for key, editor in self.editors.items():
                editor.configure(state="normal")
                editor.delete("1.0", "end")
                editor.insert("1.0", document.sections[key] if document else "")
                editor.edit_reset()
                editor.edit_modified(False)
            self.evidence.delete(*self.evidence.get_children())
            self._evidence_rows = {}
            if document:
                for item in document.evidence:
                    row = self.evidence.insert("", "end", values=(SECTIONS[item.section], item.quote.replace("\n", " ")))
                    self._evidence_rows[row] = item
                messages = list(document.warnings)
                if not document.evidence:
                    messages.insert(0, "Сведения не выделены. Проверьте транскрипцию и заполните разделы вручную.")
                self.notice.set(" ".join(messages) if messages else
                                "Выделенные сведения — черновик. Цитаты сохраняют источник; правки разделов не меняют транскрипцию.")
            else:
                self.notice.set("Выделенные сведения — черновик. Проверьте их по исходной записи.")
        finally:
            self._loading = False
        self._apply_state()

    def show_evidence(self, event: tk.Event | None = None) -> None:
        selected = self.evidence.selection()
        if not selected or selected[0] not in self._evidence_rows:
            return
        item = self._evidence_rows[selected[0]]
        self.transcript.tag_remove("evidence", "1.0", "end")
        # Ссылки в Document проверены по исходному тексту, который здесь не редактируется.
        start, end = f"1.0+{item.start}c", f"1.0+{item.end}c"
        self.transcript.tag_add("evidence", start, end)
        self.transcript.see(start)

    def new_session(self) -> None:
        if not self._idle() or not self._confirm_unsaved():
            return
        self.session.new(force=True)
        self.audio_path = None
        self.session_path = None
        self.source.set("Аудиофайл не выбран")
        self._render_document(None)
        self.status.set("Новый сеанс. Выберите аудиофайл или начните запись.")

    def import_audio(self) -> None:
        if not self._idle():
            return
        selected = filedialog.askopenfilename(title="Выберите аудиозапись диалога", parent=self.root,
            filetypes=[("Аудиозаписи", "*.wav *.mp3 *.m4a *.flac *.ogg *.mp4"), ("Все файлы", "*.*")])
        if not selected or not self._confirm_unsaved():
            return
        self.session.new(force=True)
        self.audio_path = Path(selected)
        self.session_path = None
        self.source.set("Аудио: " + str(self.audio_path))
        self._render_document(None)
        self.status.set("Файл выбран. Нажмите «Обработать аудио».")

    def process_audio(self) -> None:
        if not self._idle() or self.audio_path is None or not self._confirm_unsaved():
            return
        path = self.audio_path
        self._sync_edits()

        def operation() -> Document:
            self.runtime.ensure_server()
            self._events.put(("status", "Распознавание речи и выделение сведений. Дождитесь завершения…"))
            return self.session.process_file(path)

        def done(document: Document) -> None:
            self.session_path = None
            self._render_document(document)
            self.status.set("Готово. Проверьте текст, при необходимости исправьте разделы и сохраните сеанс.")

        self._submit("Подготовка локальной модели. При первом запуске потребуется время…", operation, done)

    def refresh_devices(self) -> None:
        if not self._idle():
            return

        def done(devices: list[dict[str, Any]]) -> None:
            self._device_map = {"Системный микрофон": None}
            default_name = "Системный микрофон"
            for item in devices:
                title = f"{item['index']}: {item['name']}" + (" (по умолчанию)" if item.get("default") else "")
                self._device_map[title] = item["index"]
                if item.get("default"):
                    default_name = title
            self.device_combo.configure(values=list(self._device_map))
            self.device_name.set(default_name)
            self.status.set(f"Найдено устройств записи: {len(devices)}." if devices else
                            "Микрофоны не найдены. Подключите устройство и обновите список.")

        self._submit("Получение списка микрофонов…", self.recorder.devices, done)

    def start_recording(self) -> None:
        if not self._idle() or not self._confirm_unsaved():
            return
        device = self._device_map.get(self.device_name.get())

        def done(result: Any) -> None:
            self.session.new(force=True)
            self.audio_path = None
            self.session_path = None
            self._render_document(None)
            self.source.set("Идёт запись с выбранного микрофона")
            self._record_started = time.monotonic()
            self._record_tick()

        self._submit("Включение выбранного микрофона…", lambda: self.recorder.start(device), done)

    def _record_tick(self) -> None:
        if self._closing or self.recorder.status != "recording":
            return
        seconds = int(time.monotonic() - self._record_started)
        if not self._busy:
            self.status.set(f"Запись {seconds // 60:02d}:{seconds % 60:02d}. Нажмите «Остановить и сохранить».")
        self.root.after(1000, self._record_tick)

    def stop_recording(self) -> None:
        if self._busy or not self._record_pending():
            return
        if self.recorder.status == "recording":
            self._submit("Остановка микрофона…", self.recorder.capture,
                         lambda result: self._choose_record_path())
        else:
            self._choose_record_path()

    def _choose_record_path(self) -> None:
        # Захват уже остановлен. Отмена выбора пути сохраняет буфер для повтора.
        self.status.set("Запись остановлена. Выберите место сохранения или отмените запись.")
        self._apply_state()
        directory = self.project_root / "user_data" / "recordings"
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._error(AppError("Не удалось подготовить папку записей. Проверьте права доступа."))
            return
        selected = filedialog.asksaveasfilename(title="Сохранить запись WAV", parent=self.root,
            initialdir=str(directory), initialfile=datetime.now().strftime("dialogue-%Y%m%d-%H%M%S.wav"),
            defaultextension=".wav", filetypes=[("Запись WAV", "*.wav")])
        if not selected:
            return

        def done(path: Path) -> None:
            self.audio_path = path
            self.source.set("Запись: " + str(path))
            warnings = self.recorder.warnings
            if warnings:
                self.notice.set("Замечания устройства записи: " + "; ".join(warnings))
            self.status.set("Запись сохранена. Нажмите «Обработать аудио».")

        self._submit("Остановка микрофона и сохранение WAV…", lambda: self.recorder.stop(selected), done)

    def discard_recording(self) -> None:
        if self._busy or not self._record_pending():
            return
        if not messagebox.askyesno("Отменить запись", "Удалить текущую несохранённую запись?", parent=self.root):
            return

        def done(result: Any) -> None:
            self.source.set("Аудиофайл не выбран")
            self.status.set("Запись отменена. Можно начать новую или загрузить файл.")

        self._submit("Остановка записи…", self.recorder.discard, done)

    def save_session(self, save_as: bool = False) -> bool:
        if not self._idle() or self.session.document is None:
            return False
        self._sync_edits()
        path = self.session_path
        if path is None or save_as:
            selected = filedialog.asksaveasfilename(title="Сохранить сеанс с правками и источниками", parent=self.root,
                initialfile="dialogue-session.json", defaultextension=".json", filetypes=[("Сеанс JSON", "*.json")])
            if not selected:
                return False
            path = Path(selected)
        try:
            self.session.save(path)
        except Exception as exc:
            self._error(exc)
            return False
        self.session_path = path
        self._update_title()
        self.status.set("Сеанс сохранён: " + str(path))
        return True

    def open_session(self) -> None:
        if not self._idle():
            return
        selected = filedialog.askopenfilename(title="Открыть сохранённый сеанс", parent=self.root,
            filetypes=[("Сеанс JSON", "*.json"), ("Все файлы", "*.*")])
        if not selected or not self._confirm_unsaved():
            return
        try:
            document = self.session.open(selected)
        except Exception as exc:
            self._error(exc)
            return
        self.session_path = Path(selected)
        source_path = Path(document.source) if document.source else None
        self.audio_path = source_path if source_path is not None and source_path.is_file() else None
        self.source.set("Сеанс: " + str(self.session_path) + (" • Аудио: " + document.source if document.source else ""))
        self._render_document(document)
        self.status.set("Сеанс открыт. Сохранённые правки восстановлены.")

    def export_current(self) -> None:
        if not self._idle() or self.session.document is None:
            return
        self._sync_edits()
        selected = filedialog.asksaveasfilename(title="Экспорт текущего текста", parent=self.root,
            initialfile="dialogue-result.txt", defaultextension=".txt", filetypes=[("Текст UTF-8", "*.txt")])
        if not selected:
            return
        try:
            export_text(self.session.document, selected)
        except Exception as exc:
            self._error(exc)
            return
        self.status.set("Текст экспортирован. Для восстановления правок и цитат сохраните также сеанс JSON.")
        self._update_title()

    def close(self) -> None:
        if self._closing:
            return
        if self._busy:
            if messagebox.askyesno("Операция ещё выполняется",
                "Закрыть окно после завершения текущей операции?\n"
                "Перед закрытием можно будет сохранить полученный результат.", parent=self.root):
                self._close_when_done = True
                self.status.set("Дождитесь завершения операции. Затем приложение предложит сохранить результат и закроется.")
            return
        if self._record_pending():
            answer = messagebox.askyesnocancel("Есть несохранённая запись",
                "Сохранить запись перед закрытием?\n«Нет» — удалить несохранённый звук и закрыть окно.", parent=self.root)
            if answer is None:
                return
            if answer:
                self.stop_recording()
                self._close_when_done = self._busy
                return
            try:
                self.recorder.discard()
            except Exception as exc:
                self._error(exc)
                return
        if not self._confirm_unsaved():
            return
        self._closing = True
        self._apply_state()
        self.root.destroy()
        # Только собственный llama-server; чужой процесс LocalRuntime не завершает.
        self.runtime.shutdown()


MedicalAudioApp = AudioApp


def main() -> None:
    """python -m audio_mvp: создание окна не загружает модели и не пишет звук."""
    import argparse
    parser = argparse.ArgumentParser(description="Локальный учебный аудиомодуль ВКР")
    parser.add_argument("--runtime-path", type=Path, default=None, help="Папка подготовленных моделей и llama.cpp")
    parser.add_argument("--asr-model", type=Path, default=None, help="Папка локальной модели faster-whisper")
    parser.add_argument("--port", type=int, default=8081, help="Локальный порт llama-server")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("порт должен находиться в диапазоне 1–65535")
    root = tk.Tk()
    AudioApp(root, runtime_root=args.runtime_path, asr_model=args.asr_model,
             port=args.port, device=args.device, compute_type=args.compute_type)
    root.mainloop()
