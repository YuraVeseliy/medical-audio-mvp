"""Регрессия D01: выбор пути после остановки внешнего микрофона."""
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

from audio_mvp.session import Session
from test_transcription import Recognizer


class RecordingTests(unittest.TestCase):
    def test_t10_cancel_save_dialog_keeps_recording_stopped(self):
        import tkinter as tk
        from audio_mvp.gui import AudioApp
        root = tk.Tk()
        root.withdraw()
        folder = Path('tmp') / ('record-probe-' + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        class Recorder:
            status = 'recording'
            warnings = []
            def capture(self):
                self.status = 'captured'
            def stop(self, path):
                raise AssertionError('Сохранение отменено')
        recorder = Recorder()
        observed = {}
        def dialog(**kwargs):
            observed['state_at_dialog'] = recorder.status
            return ''
        try:
            app = AudioApp(root, project_root=folder, session=Session(Recognizer()), recorder=recorder)
            with patch('audio_mvp.gui.filedialog.asksaveasfilename', side_effect=dialog):
                app.stop_recording()
                deadline = time.monotonic() + 3
                while 'state_at_dialog' not in observed and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.005)
            self.assertEqual(observed.get('state_at_dialog'), 'captured')
            self.assertEqual(recorder.status, 'captured')
        finally:
            root.destroy()
            shutil.rmtree(folder)
