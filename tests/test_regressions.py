"""Проверки реальных дефектов, обнаруженных рецензированием."""
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid
from types import SimpleNamespace

from audio_mvp.session import Session
from audio_mvp.model import AppError, Segment
from test_session import Recognizer, Extractor


class RegressionTests(unittest.TestCase):
    def test_d04_extraction_failure_keeps_new_transcript_visible_and_saveable(self):
        source = Path(__file__).parent / "fixtures" / "float32.wav"
        session = Session(Recognizer(), Extractor())
        previous = session.process_file(source)

        class NewRecognizer:
            def transcribe(self, path):
                return [Segment(0, 2, "Новый диалог: болит левое плечо.")]

        class FailingExtractor:
            def extract(self, text):
                raise AppError("Локальная модель превысила время ожидания.")

        session.recognizer = NewRecognizer()
        session.extractor = FailingExtractor()
        with self.assertRaises(AppError) as failure:
            session.process_file(source)
        partial = session.document
        self.assertIsNot(partial, previous)
        self.assertEqual(partial.transcript, "Новый диалог: болит левое плечо.")
        self.assertEqual(partial.source, str(source))
        self.assertEqual(partial.sections, {"complaints": "", "anamnesis": ""})
        self.assertEqual(partial.evidence, [])
        self.assertTrue(partial.warnings)
        self.assertEqual(session.state, "error")
        self.assertTrue(session.dirty)

        import tkinter as tk
        from audio_mvp.gui import AudioApp
        root = tk.Tk()
        root.withdraw()
        folder = Path("tmp") / ("partial-probe-" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        try:
            app = AudioApp(root, project_root=folder, session=session,
                           runtime=SimpleNamespace(base_url="http://127.0.0.1:8081", shutdown=lambda: None))
            app._render_document(previous)
            app.session_path = folder / "previous.json"
            with patch("audio_mvp.gui.messagebox.showerror"):
                app._error(failure.exception)
            self.assertEqual(app.transcript.get("1.0", "end-1c"), partial.transcript)
            self.assertIsNone(app.session_path)
            session.edit("complaints", "Болит левое плечо.")
            saved = folder / "partial.json"
            session.save(saved)
            restored = Session(Recognizer(), Extractor()).open(saved)
            self.assertEqual(restored.transcript, partial.transcript)
            self.assertEqual(restored.sections["complaints"], "Болит левое плечо.")
            self.assertEqual(restored.warnings, partial.warnings)
        finally:
            root.destroy()
            shutil.rmtree(folder)

    def test_d02_valid_float_wav_is_supported(self):
        source = Path(__file__).parent / "fixtures" / "float32.wav"
        document = Session(Recognizer(), Extractor()).process_file(source)
        self.assertEqual(document.source, str(source))

    def test_d01_cancel_save_dialog_keeps_stopped_recording(self):
        import tkinter as tk
        from audio_mvp.gui import AudioApp
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Нет оконной среды Tk: {exc}")
        root.withdraw()
        folder = Path("tmp") / ("record-probe-" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        class Recorder:
            status = "recording"
            warnings = []
            def capture(self):
                self.status = "captured"
            def stop(self, path):
                raise AssertionError("При отмене диалога сохранение не вызывается")
        recorder = Recorder()
        observed = {}
        def dialog(**kwargs):
            observed["state_at_dialog"] = recorder.status
            return ""
        try:
            app = AudioApp(root, project_root=folder,
                           session=Session(Recognizer(), Extractor()), recorder=recorder,
                           runtime=SimpleNamespace(base_url="http://127.0.0.1:8081", shutdown=lambda: None))
            with patch("audio_mvp.gui.filedialog.asksaveasfilename", side_effect=dialog):
                app.stop_recording()
                deadline = time.monotonic() + 3
                while "state_at_dialog" not in observed and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.005)
            self.assertEqual(observed.get("state_at_dialog"), "captured")
            self.assertEqual(recorder.status, "captured")
        finally:
            root.destroy()
            shutil.rmtree(folder)
