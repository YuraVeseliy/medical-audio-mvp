"""Проверки публичного поведения; внешние модели заменены известными ответами."""
import shutil
import unittest
import uuid
import wave
import threading
from pathlib import Path

from audio_mvp.model import AppError, Evidence, Segment
from audio_mvp.session import Session


class Recognizer:
    def transcribe(self, path):
        return [Segment(0, 1, "Болит колено."), Segment(1, 2, "Началось вчера.")]


class Extractor:
    def extract(self, text):
        return [Evidence("complaints", "Болит колено.", 0, 13),
                Evidence("anamnesis", "Началось вчера.", 14, 29)], []


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("tmp") / ("test-" + uuid.uuid4().hex)
        self.tmp.mkdir(parents=True)
        self.audio = self.tmp / "dialogue.wav"
        with wave.open(str(self.audio), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
            out.writeframes(b"\x00\x00" * 160)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_a01_supported_audio_produces_document(self):
        session = Session(Recognizer(), Extractor())
        result = session.process_file(self.audio)
        self.assertEqual((session.state, result.source), ("result", str(self.audio)))
        self.assertEqual(result.transcript, "Болит колено. Началось вчера.")
        self.assertEqual(result.sections, {"complaints": "Болит колено.", "anamnesis": "Началось вчера."})

    def test_a02_damaged_audio_has_readable_error(self):
        self.audio.write_bytes(b"not a wave")
        with self.assertRaisesRegex(AppError, "аудиофайл"):
            Session(Recognizer(), Extractor()).process_file(self.audio)

    def test_a03_no_recognized_speech_is_not_a_completed_form(self):
        class SilentRecognizer:
            def transcribe(self, path):
                return []
        with self.assertRaisesRegex(AppError, "Речь"):
            Session(SilentRecognizer(), Extractor()).process_file(self.audio)

    def test_a04_model_failure_allows_next_attempt(self):
        class FailingOnceRecognizer(Recognizer):
            failed = False
            def transcribe(self, path):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("external model failed")
                return super().transcribe(path)
        session = Session(FailingOnceRecognizer(), Extractor())
        with self.assertRaises(AppError):
            session.process_file(self.audio)
        self.assertEqual(session.state, "error")
        self.assertEqual(session.process_file(self.audio).transcript, "Болит колено. Началось вчера.")

    def test_a05_edit_changes_only_selected_section(self):
        session = Session(Recognizer(), Extractor())
        session.process_file(self.audio)
        session.edit("complaints", "Боль в левом колене.")
        self.assertEqual(session.document.sections,
                         {"complaints": "Боль в левом колене.", "anamnesis": "Началось вчера."})
        self.assertTrue(session.dirty)

    def test_a06_saved_session_restores_current_edits(self):
        session = Session(Recognizer(), Extractor())
        session.process_file(self.audio)
        session.edit("complaints", "Исправленный текст: левое колено.")
        saved = self.tmp / "осмотр.json"
        session.save(saved)
        restored = Session(Recognizer(), Extractor())
        restored.open(saved)
        self.assertEqual(restored.document.sections["complaints"], "Исправленный текст: левое колено.")
        self.assertEqual(restored.document.transcript, "Болит колено. Началось вчера.")
        self.assertFalse(restored.dirty)

    def test_a07_new_session_requires_decision_about_edits(self):
        session = Session(Recognizer(), Extractor())
        session.process_file(self.audio)
        session.edit("complaints", "Правка пользователя")
        with self.assertRaisesRegex(AppError, "правки"):
            session.new()
        session.new(force=True)
        self.assertEqual((session.state, session.document, session.dirty), ("ready", None, False))

    def test_a08_second_job_is_rejected_while_processing(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingRecognizer(Recognizer):
            def transcribe(self, path):
                entered.set()
                release.wait(5)
                return super().transcribe(path)
        session = Session(WaitingRecognizer(), Extractor())
        errors = []
        def run():
            try:
                session.process_file(self.audio)
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaisesRegex(AppError, "операции"):
                session.process_file(self.audio)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
