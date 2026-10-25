# Компоненты учебного комплекта

Приложение написано на Python. Веса моделей не обучались и не изменялись авторами. В portable-комплект включаются исходные лицензионные уведомления установленных библиотек, Python и моделей; точные версии и хеши — в манифестах.

| Компонент | Роль | Источник и условия |
|---|---|---|
| Qwen2.5-3B-Instruct GGUF | Выбор номеров исходных предложений | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF — Qwen Research License, некоммерческое исследование/оценка; полный текст и Notice.txt приложены |
| faster-whisper / Whisper small | Распознавание русской речи | https://github.com/SYSTRAN/faster-whisper и https://github.com/openai/whisper — MIT |
| CTranslate2 | Выполнение ASR на CPU | https://github.com/OpenNMT/CTranslate2 — MIT |
| llama.cpp | Выполнение GGUF на CPU | https://github.com/ggml-org/llama.cpp — MIT |
| Tkinter / Tcl/Tk | Окно приложения | https://www.tcl.tk/software/tcltk/license.html |
| PyAV / FFmpeg | Декодирование аудио | https://github.com/PyAV-Org/PyAV и https://ffmpeg.org/legal.html; условия конкретных бинарных библиотек приложены к сборке |
| sounddevice / PortAudio | Входной поток микрофона | https://github.com/spatialaudio/python-sounddevice и https://www.portaudio.com/license.html |
| soundfile / libsndfile | Запись WAV | https://github.com/bastibe/python-soundfile и https://libsndfile.github.io/libsndfile/; лицензии бинарных библиотек сохранены |
| ONNX Runtime, NumPy, tokenizers | Поддержка вычислений, VAD и токенизация | Лицензии установленных дистрибутивов в licenses/python-packages/ |
| Python | Среда portable | PSF License, копия в licenses/ |
| PyInstaller | Сборка EXE | https://pyinstaller.org/en/stable/license.html — GPL с исключением для создаваемого bootloader-комплекта |

Piper используется только для подготовки синтетического технического набора. Голос ru_RU-irina-medium основан на наборе, лицензия которого в model card обозначена Unknown. Голосовая модель не включается в portable/исходный архив; её карточка и происхождение учебных WAV сохранены в data/tts-source/. Права на клиническое или коммерческое использование этого голоса данным экспериментом не подтверждаются.

Open Medical Scribe и Buzz используются как внешние аналоги в исследовательском окружении. Их исходники не изменялись и не включаются в рабочий EXE. Репозитории, версии, исходные файлы компонентов и лицензии перечислены в экспериментах.
