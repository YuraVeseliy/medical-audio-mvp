"""Публичный сеанс транскрибации; подменяется только внешний ASR."""
from pathlib import Path
import shutil
import unittest
import uuid
import wave
import threading

from audio_mvp.model import Segment, AppError
from audio_mvp.session import Session
from audio_mvp.storage import export_text


class Recognizer:
    model_id = 'external-test-asr'

    def transcribe(self, path):
        yield Segment(0, 1, 'Что вас беспокоит?')
        yield Segment(1, 2, 'Болит левое плечо.')
        yield Segment(2, 3, 'Боль появилась вчера.')


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path('tmp') / ('asr-test-' + uuid.uuid4().hex)
        self.tmp.mkdir(parents=True)
        self.audio = self.tmp / 'dialogue.wav'
        with wave.open(str(self.audio), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b'\x00\x00' * 160)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_t01_complete_dialogue_without_extractor(self):
        session = Session(Recognizer())
        result = session.process_file(self.audio)
        self.assertEqual(result.transcript, 'Что вас беспокоит? Болит левое плечо. Боль появилась вчера.')
        self.assertEqual(result.text, result.transcript)
        self.assertEqual(result.model_id, 'external-test-asr')
        self.assertEqual(len(result.segments), 3)
        self.assertEqual(result.source, str(self.audio))
        self.assertEqual(session.state, 'result')

    def test_t02_edit_save_open_preserves_original_and_current_text(self):
        session = Session(Recognizer())
        original = session.process_file(self.audio).transcript
        session.edit('Болит левое плечо. Боль появилась вчера вечером.')
        self.assertTrue(session.dirty)
        saved = self.tmp / 'session.json'
        session.save(saved)
        restored = Session(Recognizer())
        document = restored.open(saved)
        self.assertEqual(document.transcript, original)
        self.assertEqual(document.text, 'Болит левое плечо. Боль появилась вчера вечером.')
        self.assertEqual(document.model_id, 'external-test-asr')
        self.assertEqual(len(document.segments), 3)
        self.assertFalse(restored.dirty)
        exported = self.tmp / 'result.txt'
        export_text(document, exported)
        self.assertEqual(exported.read_text(encoding='utf-8'), document.text + '\n')

    def test_t03_damaged_audio_is_rejected(self):
        self.audio.write_bytes(b'not an audio file')
        with self.assertRaisesRegex(AppError, 'аудиофайл'):
            Session(Recognizer()).process_file(self.audio)

    def test_t04_empty_recognition_is_reported(self):
        class SilentRecognizer:
            def transcribe(self, path):
                return []
        session = Session(SilentRecognizer())
        with self.assertRaisesRegex(AppError, 'Речь не распознана'):
            session.process_file(self.audio)
        self.assertIsNone(session.document)

    def test_t05_failure_allows_retry(self):
        class FailingOnce(Recognizer):
            failed = False
            def transcribe(self, path):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError('external ASR failure')
                return super().transcribe(path)
        session = Session(FailingOnce())
        with self.assertRaises(AppError):
            session.process_file(self.audio)
        self.assertEqual(session.state, 'error')
        self.assertIn('Болит левое плечо.', session.process_file(self.audio).transcript)

    def test_t06_new_session_protects_unsaved_text(self):
        session = Session(Recognizer())
        session.process_file(self.audio)
        session.edit('Правка пользователя')
        with self.assertRaisesRegex(AppError, 'правки'):
            session.new()
        self.assertEqual(session.document.text, 'Правка пользователя')
        session.new(force=True)
        self.assertEqual((session.state, session.document, session.dirty), ('ready', None, False))

    def test_t07_second_job_is_rejected_while_processing(self):
        entered, release = threading.Event(), threading.Event()
        class Waiting(Recognizer):
            def transcribe(self, path):
                entered.set()
                release.wait(5)
                return super().transcribe(path)
        session = Session(Waiting())
        errors = []
        def run():
            try:
                session.process_file(self.audio)
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaisesRegex(AppError, 'операции'):
                session.process_file(self.audio)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])

    def test_t08_corrupt_session_does_not_replace_open_result(self):
        session = Session(Recognizer())
        document = session.process_file(self.audio)
        session.save(self.tmp / 'good.json')
        damaged = self.tmp / 'broken.json'
        damaged.write_text('{bad json', encoding='utf-8')
        with self.assertRaisesRegex(AppError, 'повреждён'):
            session.open(damaged)
        self.assertIs(session.document, document)

    def test_t09_float_wav_can_be_transcribed(self):
        source = Path(__file__).parent / 'fixtures/float32.wav'
        self.assertEqual(Session(Recognizer()).process_file(source).source, str(source))

    def test_t11_long_audio_reaches_final_sample_without_gaps(self):
        import numpy as np
        from audio_mvp.providers import ASRRecognizer
        samples = np.zeros(384000, dtype='<i2')
        samples[-1] = 30000
        with wave.open(str(self.audio), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(samples.tobytes())
        class ExternalSampleReader:
            def recognize(self, waveform):
                return f'Последний отсчёт {int(round(float(waveform[-1]) * 32768))}.'
        document = Session(ASRRecognizer(backend=ExternalSampleReader())).process_file(self.audio)
        self.assertTrue(document.transcript.endswith('30000.'))
        self.assertEqual(document.metrics['sample_count'], 384000)
        boundaries = document.metrics['chunks']
        self.assertEqual(boundaries[0]['start_sample'], 0)
        self.assertEqual(boundaries[-1]['end_sample'], 384000)
        self.assertGreater(len(boundaries), 1)
        for previous, current in zip(boundaries, boundaries[1:]):
            self.assertEqual(previous['end_sample'], current['start_sample'])
        self.assertTrue(all(item['end_sample'] - item['start_sample'] <= 320000 for item in boundaries))
