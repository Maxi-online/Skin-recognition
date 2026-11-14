"""
Главный entry point для запуска FastAPI сервиса.
"""

import os
import sys
import json
from pathlib import Path
import subprocess

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent
    profile_path = project_root / 'config' / 'research_profile.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8')) if profile_path.exists() else {}
    backend = os.getenv('SKIN_BACKEND', profile.get('backend', 'v2'))
    interpreter = project_root / ('.venv-gpu' if backend in ('v4', 'v42') else '.venv') / 'Scripts' / 'python.exe'
    if not interpreter.is_file():
        raise SystemExit(f'Required {backend} environment is missing: {interpreter}')
    if Path(sys.executable).resolve() != interpreter.resolve():
        raise SystemExit(subprocess.run([str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
                                       cwd=str(project_root)).returncode)

import uvicorn
from src.api.app import app, init_recognition_system

if __name__ == '__main__':
    # --- Windows/PowerShell: фикс "кракозябр" в выводе (кодировка консоли) ---
    # Cursor/IDE обычно читает stdout как UTF-8, а Windows может писать CP866/CP1251.
    # Принудительно переводим stdout/stderr в UTF-8, чтобы русские сообщения читались корректно.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Параметры из переменных окружения
    port = int(os.getenv('PORT', 8001))
    host = os.getenv('HOST', '127.0.0.1')
    eager_init = str(os.getenv("EAGER_INIT", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
    
    print("=" * 60)
    print("Запуск исследовательского сервера распознавания")
    print(f"Backend: {backend}; режим указан в веб-интерфейсе")
    print("=" * 60)
    # ВАЖНО: 0.0.0.0 — это адрес прослушивания (bind), его нельзя открывать в браузере.
    # Для браузера используйте localhost/127.0.0.1.
    print(f"\nBind:           http://{host}:{port} (служебное, не для браузера)")
    print(f"Веб-интерфейс:  http://localhost:{port}/")
    print(f"Health:         http://localhost:{port}/health")
    print(f"API docs:       http://localhost:{port}/docs")
    if eager_init:
        print("\nEAGER_INIT=1: пробуем инициализировать систему распознавания при старте...")
    else:
        print("\nИнициализация системы распознавания произойдет при первом запросе... (EAGER_INIT=0)")
    print("=" * 60)
    print()
    
    # По умолчанию запускаем веб-сервер без блокирующей инициализации,
    # чтобы UI/health/docs были доступны сразу даже при тяжелых зависимостях/моделях.
    if eager_init:
        try:
            print("Инициализация системы распознавания...")
            init_recognition_system()
            print("Система готова к работе.")
        except Exception as e:
            print(f"Предупреждение: Ошибка при инициализации системы: {e}")
            print("Система будет инициализирована при первом запросе.")
            import traceback
            traceback.print_exc()
    
    uvicorn.run(app, host=host, port=port, log_level="info")

