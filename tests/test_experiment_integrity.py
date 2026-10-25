"""Публичные проверки пригодности набора и сохранности эксперимента."""
import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid
import wave


class ExperimentIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path('tmp') / ('integrity-' + uuid.uuid4().hex)
        self.tmp.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_t13_duplicate_audio_cannot_cross_evaluation_groups(self):
        from scripts.compare_asr import validate_dataset
        audio = self.tmp / 'audio.wav'
        with wave.open(str(audio), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b'\0\0' * 160)
        reference = self.tmp / 'ref.txt'
        reference.write_text('Болит рука.', encoding='utf-8')
        item = {'prepared': 'audio.wav', 'prepared_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
                'reference': 'ref.txt', 'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest()}
        manifest = {'reference_verification': {'status': 'confirmed_by_recording_author'},
                    'duplicate_decoded_audio': [['dev01','control01']],
                    'items': [{**item,'id':'dev01','split':'development'},
                              {**item,'id':'control01','split':'final_control'}]}
        (self.tmp/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'дубликат|одинаков'):
            validate_dataset(self.tmp)
        manifest['duplicate_decoded_audio'] = []
        (self.tmp/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'дубликат|одинаков'):
            validate_dataset(self.tmp)

    def test_t14_control_requires_same_settings_and_weights(self):
        from scripts.compare_asr import validate_control
        previous = {'split':'selection','chosen_model':'whisper-small','source_sha256':'source-A',
                    'dataset_manifest_sha256':'data-A','threads_requested':8,
                    'models':[{'model_id':'whisper-small','installation_fingerprint':'weights-A'}]}
        installation = {'model_id':'whisper-small','fingerprint':'weights-A'}
        self.assertEqual(validate_control(previous,'source-A','data-A',8,installation),'whisper-small')
        with self.assertRaisesRegex(ValueError, 'настройки'):
            validate_control(previous,'source-A','data-A',1,installation)
        with self.assertRaisesRegex(ValueError, 'веса'):
            validate_control(previous,'source-A','data-A',8,{**installation,'fingerprint':'weights-B'})

    def test_t15_interrupted_worker_keeps_completed_measurement(self):
        from scripts.compare_asr import recover_worker_result, aggregate
        row = {'id':'dialog01','successful':True,'errors':1,'reference_words':2,'seconds':1.0,'rtf':0.1}
        (self.tmp/'rows.json').write_text(json.dumps([row]),encoding='utf-8')
        (self.tmp/'checkpoint.json').write_text(json.dumps({'model_load_seconds':0.5,
            'peak_worker_rss_bytes':1000,'last_completed_count':1}),encoding='utf-8')
        recovered = recover_worker_result(self.tmp,'whisper-small')
        self.assertEqual(recovered['rows'],[row])
        self.assertEqual(recovered['peak_worker_rss_bytes'],1000)
        summary = aggregate(recovered,2,4)
        self.assertEqual((summary['successful'],summary['errors'],summary['reference_words'],summary['wer']),
                         (1,3,4,0.75))
        self.assertFalse(summary['eligible'])
        self.assertFalse(summary['memory_measurement_complete'])
