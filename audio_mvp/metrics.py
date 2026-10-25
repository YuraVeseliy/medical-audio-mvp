"""Строгий WER по одной общей нормализации без словаря медицинских исправлений."""
import re


def words(text):
    return re.findall(r'\w+', text.casefold().replace('ё', 'е'), re.UNICODE)


def word_errors(reference, hypothesis):
    expected, actual = words(reference), words(hypothesis)
    if not expected:
        raise ValueError('Эталон для WER не должен быть пустым')
    costs = [[0] * (len(actual) + 1) for _ in range(len(expected) + 1)]
    for i in range(len(expected) + 1):
        costs[i][0] = i
    for j in range(len(actual) + 1):
        costs[0][j] = j
    for i, ref_word in enumerate(expected, 1):
        for j, hyp_word in enumerate(actual, 1):
            costs[i][j] = min(costs[i - 1][j] + 1, costs[i][j - 1] + 1,
                              costs[i - 1][j - 1] + (ref_word != hyp_word))
    substitutions = deletions = insertions = 0
    i, j = len(expected), len(actual)
    while i or j:
        if i and j and expected[i - 1] == actual[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i and j and costs[i][j] == costs[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    errors = substitutions + deletions + insertions
    return {'errors': errors, 'reference_words': len(expected), 'substitutions': substitutions,
            'deletions': deletions, 'insertions': insertions, 'wer': errors / len(expected)}
