"""Сохранить уведомления установленных зависимостей рядом с комплектом."""
from importlib.metadata import distributions
import json
from pathlib import Path
import re
import shutil
import sys

def collect(destination):
    target = Path(destination)
    target.mkdir(parents=True,exist_ok=True)
    records=[]
    for dist in distributions():
        name=re.sub(r'[^A-Za-z0-9_.-]','_',dist.metadata['Name'])
        for item in dist.files or []:
            basename=Path(str(item)).name
            if not any(word in basename.lower() for word in ['license','copying','notice']):
                continue
            source=Path(dist.locate_file(item))
            if not source.is_file() or source.stat().st_size > 5*1024*1024:
                continue
            # Каждое происхождение имеет собственную папку, одноимённые LICENSE не затираются.
            relative=Path(name)/re.sub(r'[^A-Za-z0-9_.-]','_',str(item))
            output=target/'python-packages'/relative
            output.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,output)
            records.append({'distribution':name,'version':dist.version,'source':str(item),'copied':str(relative)})
    python_license=Path(sys.base_prefix)/'LICENSE.txt'
    if python_license.is_file():shutil.copy2(python_license,target/'Python-LICENSE.txt')
    (target/'collected-notices.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Уведомлений: {len(records)}')

if __name__=='__main__':
    collect(sys.argv[1])
