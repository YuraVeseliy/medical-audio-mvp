"""Сводка по сохранённым попыткам; не вызывает и не перенастраивает модели."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]

def load(label):
    path=ROOT/'reports'/label/'results.json'
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    labels={'Наш IndexedExtractor':'our','OMS API':'oms','Buzz AI Summary':'buzz'}
    text=['\n### Формирование разделов на эталонном тексте B1.1\n',
          '| Конфигурация с общей 3B-моделью | Завершено | TP / прогнозов | Точность | Полнота всех 72 | Медиана, с | Диапазон, с |',
          '|---|---|---|---|---|---|---|']
    records=[]
    for name,label in labels.items():
        data=load('comparison-final-b11-'+label);s=data['summary'];t=s['seconds']
        expected_all=4*s['attempted'];tp=s['true_positive'];pred=s['predicted']
        text.append(f'| {name} | {s["completed"]}/{s["attempted"]} | {tp}/{pred} | {100*tp/pred:.2f}% | {100*tp/expected_all:.2f}% | {t["median"]:.3f} | {t["min"]:.3f}–{t["max"]:.3f} |')
        records.append({'configuration':name,**s,'FP':pred-tp,'FN_all_attempts':expected_all-tp,
                        'expected_all_attempts':expected_all,'raw_results':'reports/comparison-final-b11-'+label+'/results.json'})
    pipeline=load('comparison-final-b11-pipeline');s=pipeline['summary'];runs=pipeline['runs']
    counts={k:sum(r['wer_metrics'][k] for r in runs if r['status']=='completed') for k in ['errors','reference_words','substitutions','deletions','insertions']}
    rtf=statistics.median(r['rtf'] for r in runs if r['status']=='completed')
    text.extend(['',
        'Buzz получил немного лучшие строгие показатели извлечения, наша конфигурация — меньшую прогретую медиану времени. Разница в один верно найденный факт среди 72 повторных ожиданий не является доказанным общим превосходством. Все три решения ошибались. Результат нельзя приписать только приложению: общий LLM один, нативные обёртки, схема и параметры различаются.',
        '', '### Полный собственный путь WAV → Document на B1.1','',
        f'Настоящий Session.process_file завершил {s["completed"]}/{s["attempted"]} попыток. TP={s["true_positive"]}, прогнозов={s["predicted"]}, ожидаемых=72. Строгая точность {100*s["precision"]:.2f}%, полнота {100*s["recall"]:.2f}%. Медиана {s["seconds"]["median"]:.3f} с, диапазон {s["seconds"]["min"]:.3f}–{s["seconds"]["max"]:.3f} с; медиана RTF={rtf:.3f}. WER внутри этих же запусков: {counts["errors"]}/{counts["reference_words"]} = {100*counts["errors"]/counts["reference_words"]:.2f}%.',
        '',
        'Это строгая оценка совпадения целых цитат с исходным эталоном, а не утверждение, что только четверть медицинского смысла передана правильно. Даже одно изменённое слово или склеенные вопрос и ответ могут дать FP/FN. Но причина снижения не сводится к строгости метрики: есть настоящие ошибочные вопросы и неверные разделы. Метод не менялся после просмотра B1.1.',
        '', '### Три проверяемых примера ошибок','',
        '| Случай | Наблюдение | Что это показывает |',
        '|---|---|---|',
        '| B01, чистый текст | В жалобы выбран вопрос «Что ещё беспокоит?», в анамнез — «Когда была операция?» и «Как чувствовали себя до операции?» | Точное наличие в источнике не делает вопрос фактом; проблема выбора LLM существует без ошибок ASR |',
        '| B05, полная цепочка | «Пальцы на правой руке холодные.» распознано, но помещено в анамнез вместо заданных жалоб | Источник есть, раздел неверен по нашему эталону; один WER это не обнаружит |',
        '| B06, полная цепочка | WER первого повтора 1/30 = 3.33%, но ASR склеила «Сильно хочется пить, когда появилась сухость.» и «Глотать сейчас не больно, когда последний раз пили.» | Низкий WER не учитывает пунктуацию. Предложение-кандидат содержит ответ и вопрос, поэтому строгая цитата не совпадает и выбор затруднён |',
        '',
        'Исходные тексты, предсказания и списки matched/unmatched/missed: reports/comparison-final-b11-our/results.json и comparison-final-b11-pipeline/results.json, ID b01/b05/b06, repetition=1. Разбор является техническим чтением наблюдений, не независимой врачебной разметкой.',
        '', '## 6. Решение по качеству и следующий инкремент','',
        'Готов работающий учебный прототип с импортом/записью, локальной обработкой, исходником, правками и сохранением. Быстрые тесты подтверждают программное поведение, но не устраняют ошибки моделей. Автоматическое распределение в текущей конфигурации не принимается как надёжное заполнение медицинской документации. Дефект качества D05 оставлен открытым, а не объявлен исправленным благодаря зелёным unittest.',
        '',
        'Практическая польза данного ДЗ — реализованный и измеренный путь, прозрачные данные, найденные и исправленные программные ошибки и обоснованное решение о границах применения. Мы не доказали экономию времени врача, превосходство над Buzz/OMS, пригодность для больницы или отсутствие утечки на всех уровнях.',
        '',
        'Дальнейшая работа: лично прослушать и разметить вымышленные двухголосые записи; согласовать разделы и допустимые ошибки с врачом; расширить development-примеры порядками вопросов и отрицаниями; проверить более устойчивое выделение границ/фрагментов и подходящие модели на новом независимом наборе. Нельзя подбирать исправление на B1.1 и затем предъявлять те же цифры как независимую итоговую проверку.',
        '',
        'В презентации достаточно объяснить общий вход, шесть сценариев, четыре ожидаемых факта, три повтора для времени, определения WER/точности/полноты, один пример ошибки и решение по результатам. Нельзя говорить «мы победили аналоги», «77 тестов доказывают качество» или «готово к применению в реанимации».',
        '', '## 7. Как воспроизвести','',
        'Установить requirements.txt и requirements-experiments.txt с constraints requirements-lock.txt. Подготовить модели и исходные компоненты по experiments/analog-plan.md. Запустить 3B llama-server на 8082 и штатный OMS API через scripts/start_oms.mjs. Для Buzz убрать HTTP/HTTPS/ALL proxy и задать NO_PROXY=127.0.0.1,localhost,::1: первая development-попытка без этого дала 503 и сохранена отдельно. Приложение всегда обходит системный прокси для локальной модели.',
        '',
        'Наш компонент: python scripts/compare_local.py extract --manifest data/evaluation-b11/manifest.json --engine our-indexed --endpoint http://127.0.0.1:8082 --model-server http://127.0.0.1:8082 --repeat 3 --out reports/new-our-run. Полный путь: заменить extract на pipeline. Для OMS: engine=oms, --indexed, endpoint=http://127.0.0.1:8787. Для Buzz: engine=buzz, --indexed, endpoint=http://127.0.0.1:8082/v1, --buzz-source .runtime/analogs/buzz. Старый каталог результатов намеренно не перезаписывается.',
        '',
        'Закреплённая модель 3B: revision 7dabda4d13d513e3e842b20f0d435c732f172cbe; SHA256 626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d. Номера, подсказка и исходники нового кандидата зафиксированы до B1.1 в reports/comparison-final-config.json. Полные значения, без округления таблиц, сохраняются в JSON.'
    ])
    path=ROOT/'experiments/comparison-results.md'
    original=path.read_text(encoding='utf-8')
    marker='\n### Формирование разделов на эталонном тексте B1.1'
    if marker in original:original=original.split(marker)[0]
    path.write_text(original+'\n'.join(text)+'\n',encoding='utf-8')
    summary={'updated_at_utc':datetime.now(timezone.utc).isoformat(),'status':'Technical experiments completed; human M01–M06 pending; quality defect D05 open',
             'initial_A':'reports/comparison-initial-overview.json','final_B11':records,
             'pipeline':{**s,'WER_counts':counts,'RTF_median':rtf,'raw_results':'reports/comparison-final-b11-pipeline/results.json'},
             'limitations':['synthetic one voice','6 unique evaluation cases, not 18 independent patients','developer references, no physician validation','component comparison, not whole Buzz GUI','different native wrappers/settings','no tuning on final B11']}
    (ROOT/'reports/comparison-final-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
