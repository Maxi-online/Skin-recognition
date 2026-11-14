"""
Запуск local research сервиса через Uvicorn.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

if __name__ == '__main__':
    # 8000 в этом проекте часто занят; используем 8001 по умолчанию
    port = os.getenv('PORT', '8001')
    host = os.getenv('HOST', '127.0.0.1')
    workers = os.getenv('WORKERS', '1')
    if workers != '1':
        raise SystemExit('File-backed gallery requires WORKERS=1')
    root = Path(__file__).resolve().parent
    profile_path = root / 'config' / 'research_profile.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8')) if profile_path.exists() else {}
    backend = os.getenv('SKIN_BACKEND', profile.get('backend', 'v2'))
    environment = '.venv-gpu' if backend in ('v4', 'v42') else '.venv'
    interpreter = root / environment / 'Scripts' / 'python.exe'
    if not interpreter.is_file():
        raise SystemExit(f'Required {backend} environment is missing: {interpreter}')
    
    subprocess.run([
        str(interpreter), '-m', 'uvicorn',
        'src.api.app:app',
        '--host', host,
        '--port', port,
        '--workers', workers
    ], cwd=str(root), check=True)

