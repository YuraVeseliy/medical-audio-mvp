"""Независимо посчитанные примеры метрики, используемой в отчёте."""
import unittest
from audio_mvp.metrics import word_errors


class MetricsTests(unittest.TestCase):
    def test_t12_known_substitution_insertion_and_denominator(self):
        result = word_errors('У меня болит левая рука.', 'у меня сильно болит правая рука')
        self.assertEqual((result['errors'], result['reference_words'], result['substitutions'],
                          result['deletions'], result['insertions']), (2, 5, 1, 0, 1))
        self.assertEqual(result['wer'], 0.4)
