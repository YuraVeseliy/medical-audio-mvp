"""Чистый архив актуальных исходников без голосовых записей, весов и старых сравнений."""
import argparse
from pathlib import Path
import sys,zipfile,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audio_mvp.provenance import source_version

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    names=['README.md','AGENTS.md','run_app.pyw','launch.cmd','setup.cmd',
           'requirements.txt','requirements-lock.txt','requirements-docs.txt','.gitignore','.gitattributes',
           'THIRD_PARTY_NOTICES.md','docs/ASR_план_разработки_и_проверки.pdf',
           'docs/ASR_отчет_сравнение_и_тестирование.pdf']
    names += ['scripts/'+s for s in ['check.py','check_gui_scenario.py','compare_asr.py',
              'prepare_human_data.py','setup_asr_models.py','package_asr_source.py','make_pdf.py']]
    selected={ROOT/name for name in names if (ROOT/name).is_file()}
    for folder in ['audio_mvp','tests','configs','skills','licenses','.github']:
        selected.update(p for p in (ROOT/folder).rglob('*') if p.is_file()
                        and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'})
    selected.update((ROOT/'docs').glob('asr-*'))
    selected.add(ROOT/'docs/manual-tests.md')
    for folder in ['asr_checks','asr_ci','asr_gui_final','asr-tdd','asr_static','asr-comparison','defects/D01','defects/D02']:
        selected.update(p for p in (ROOT/'reports'/folder).rglob('*') if p.is_file())
    selected.add(ROOT/'reports/asr-code-review.md')
    # Синтетический демонстрационный файл, не личная запись человека.
    selected.add(ROOT/'data/audio/dev01.wav')
    selected={p for p in selected if p.is_file()}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(selected):z.write(path,'medical-audio-mvp/'+path.relative_to(ROOT).as_posix())
        z.writestr('medical-audio-mvp/BUILD_INFO.json',json.dumps(source_version(ROOT),ensure_ascii=False,indent=2))
    print(json.dumps({'archive':str(args.out),'files':len(selected)+1,'bytes':args.out.stat().st_size}))
if __name__=='__main__':main()
