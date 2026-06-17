#!/usr/bin/env python3
"""
SubsToText - приложение в трее Windows.
Клик по иконке -> вставить ссылку на YouTube -> скачать авто-субтитры,
почистить от мусора и сохранить .txt в папку "Загрузки".
После успеха: открыть папку / открыть файл / скопировать файл в буфер.

Зависимости: pip install pystray pillow
Опционально (локальная транскрибация, если нет субтитров): pip install faster-whisper
Опционально (ускорение Whisper на GPU, только NVIDIA; без них — работа на CPU):
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
В системе уже должны стоять: yt-dlp, deno (JS-движок).
"""

import os
import re
import sys
import glob
import queue
import shutil
import tempfile
import threading
import subprocess
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import ttk, messagebox

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: missing deps. Run:  pip install pystray pillow")
    sys.exit(1)

# ---------- настройки ----------
BROWSER = "firefox"               # откуда брать cookies
LANGS = "ru,en"                   # языки субтитров (доступные для выбора)
LANG_PRIORITY = ["ru", "en"]      # предпочтительный язык на выходе
LANG_NAMES = {"ru": "Русский", "en": "English"}   # подписи в интерфейсе
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
# CREATE_NO_WINDOW - чтобы не мигали чёрные окна консоли при вызове yt-dlp
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Повтор при сбоях скачивания субтитров (например HTTP 429 Too Many Requests)
SUB_RETRIES = 5        # число попыток
SUB_RETRY_SLEEP = 5    # пауза между попытками, секунд

# Сколько видео плейлиста качать параллельно (1 = последовательно).
# Больше потоков = быстрее, но выше риск HTTP 429 от YouTube.
PLAYLIST_WORKERS = 4

# Глобальная горячая клавиша: Ctrl+Alt+Y
HOTKEY_MODS = 0x0002 | 0x0001   # MOD_CONTROL | MOD_ALT
HOTKEY_VK   = 0x59               # Virtual-Key code для Y
HOTKEY_ID   = 1
# --------------------------------


# ======================================================================
#  ЛОГИКА СКАЧИВАНИЯ И ЧИСТКИ
# ======================================================================

def pick_vtt(tmp_dir: str) -> str:
    """Выбирает один .vtt по приоритету языков."""
    files = glob.glob(os.path.join(tmp_dir, "*.vtt"))
    if not files:
        raise FileNotFoundError("Субтитры не найдены (.vtt отсутствует)")
    for lang in LANG_PRIORITY:
        for f in files:
            if f".{lang}.vtt" in f:
                return f
    return files[0]


def clean_vtt(path: str) -> str:
    """Превращает .vtt в чистый связный текст без таймкодов, тегов и дублей."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    cleaned = []
    for line in lines:
        if line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if "-->" in line:                       # строки с таймкодами
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()   # inline-теги
        if not line:
            continue
        cleaned.append(line)

    # дедуп подряд идущих "осевших" строк rolling-формата
    deduped = []
    for line in cleaned:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    text = " ".join(deduped)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def _register_cuda_dlls() -> None:
    """Подгружает CUDA-библиотеки из pip-пакетов nvidia-* в текущий процесс.

    DLL (cublas64_12.dll / cudnn*.dll) лежат в site-packages\\nvidia\\...\\bin,
    которых нет в путях поиска. Одного os.add_dll_directory CTranslate2 на
    Windows недостаточно — он не находит cublas, поэтому предзагружаем ключевые
    библиотеки явно через ctypes.WinDLL. Тихо игнорируем, если пакетов нет
    (тогда сработает откат на CPU в run_whisper)."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    import sysconfig  # noqa: PLC0415

    nvidia_root = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
    if not os.path.isdir(nvidia_root):
        return
    # Сначала добавляем все bin-папки, чтобы предзагрузка нашла зависимости.
    bin_dirs = []
    for pkg in os.listdir(nvidia_root):
        bin_dir = os.path.join(nvidia_root, pkg, "bin")
        if os.path.isdir(bin_dir):
            bin_dirs.append(bin_dir)
            try:
                os.add_dll_directory(bin_dir)
            except OSError:
                pass
    # Предзагружаем cublas/cudnn по полному пути, чтобы они уже были в процессе.
    for bin_dir in bin_dirs:
        for name in ("cublas64_12.dll", "cudnn64_9.dll"):
            dll = os.path.join(bin_dir, name)
            if os.path.isfile(dll):
                try:
                    ctypes.WinDLL(dll)
                except OSError:
                    pass


def run_whisper(url: str, model_size: str, log_q: "queue.Queue") -> str:
    """Скачивает аудио и транскрибирует через faster-whisper. Возвращает путь к .txt."""
    from faster_whisper import WhisperModel  # noqa: PLC0415

    tmp_dir = tempfile.mkdtemp(prefix="ytsubs_whisper_")
    try:
        out_template = os.path.join(tmp_dir, "audio.%(ext)s")
        cmd = [
            "yt-dlp",
            *( ["--cookies-from-browser", BROWSER] if BROWSER else [] ),
            "--remote-components", "ejs:github",
            "--format", "bestaudio[ext=m4a]/bestaudio/best",
            "--no-playlist",
            "-o", out_template,
            "--newline",
            url,
        ]
        log_q.put("Скачиваем аудио...\n")
        log_q.put(3.0)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=NO_WINDOW,
        )
        if proc.stdout:
            for line in proc.stdout:
                log_q.put(line)
                dm = re.match(r"\[download\]\s+([\d.]+)%", line.strip())
                if dm:
                    log_q.put(3.0 + float(dm.group(1)) * 0.37)  # 3–40%
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("yt-dlp не смог скачать аудио")

        audio_files = glob.glob(os.path.join(tmp_dir, "audio.*"))
        if not audio_files:
            raise FileNotFoundError("Аудиофайл не найден после скачивания")
        audio_path = audio_files[0]

        log_q.put(f"Транскрибируем (модель: {model_size})...\n")
        log_q.put(42.0)
        _register_cuda_dlls()
        try:
            # cublas/cudnn подгружаются лениво на первом transcribe(), поэтому
            # пробный прогон делаем здесь же — иначе сбой GPU не поймать.
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            segments, info = model.transcribe(audio_path, beam_size=5)
            info.language  # форсируем работу, чтобы поймать ошибку загрузки CUDA
            log_q.put("Устройство: GPU (CUDA, float16)\n")
        except Exception as e:  # noqa: BLE001 — нет CUDA/драйверов/библиотек → откат на CPU
            log_q.put(f"GPU недоступен ({e}); работаем на CPU\n")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio_path, beam_size=5)
            log_q.put("Устройство: CPU (int8)\n")
        log_q.put(f"Язык: {info.language} ({info.language_probability:.0%})\n")
        log_q.put(45.0)

        duration = getattr(info, "duration", None)
        parts = []
        for seg in segments:
            parts.append(seg.text.strip())
            log_q.put(f"[{seg.start:.1f}s] {seg.text.strip()}\n")
            if duration:
                log_q.put(45.0 + min(seg.end / duration, 1.0) * 52.0)  # 45–97%

        text = re.sub(r"\s+", " ", " ".join(parts)).strip()

        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
        name = f"transcript_{m.group(1)}.txt" if m else "transcript.txt"
        os.makedirs(DOWNLOADS, exist_ok=True)
        out_path = os.path.join(DOWNLOADS, name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        log_q.put(f"Saved: {out_path}\n")
        log_q.put(100.0)
        return out_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def fetch_subtitle_text(url: str, log_q: "queue.Queue | None", progress_cb=None) -> str:
    """
    Качает субтитры и возвращает чистый текст (без сохранения на диск).
    log_q — очередь для строк/флоатов прогресса (может быть None).
    progress_cb — опциональный callable(float) для передачи прогресса 0..100.
    Бросает FileNotFoundError если субтитры не найдены, RuntimeError при других ошибках.
    """
    def _prog(pct: float):
        if log_q is not None:
            log_q.put(pct)
        if progress_cb is not None:
            progress_cb(pct)

    def _log(msg: str):
        if log_q is not None:
            log_q.put(msg)

    tmp_dir = tempfile.mkdtemp(prefix="ytsubs_")
    try:
        out_template = os.path.join(tmp_dir, "subs")
        cmd = [
            "yt-dlp",
            *( ["--cookies-from-browser", BROWSER] if BROWSER else [] ),
            "--remote-components", "ejs:github",
            "--write-auto-subs",
            "--skip-download",
            "--no-playlist",
            "--sub-langs", LANGS,
            "-o", out_template,
            "--newline",
            url,
        ]
        _log("Starting yt-dlp...\n")
        _prog(3.0)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                _log(line)
                l = line.strip()
                if "Extracting cookies" in l:
                    _prog(8.0)
                elif "Extracted" in l and "cookies" in l:
                    _prog(15.0)
                elif "Extracting URL" in l:
                    _prog(22.0)
                elif "Downloading webpage" in l:
                    _prog(35.0)
                elif "Downloading android" in l or "Downloading m3u8" in l or "Downloading MPD" in l:
                    _prog(50.0)
                elif "Downloading subtitles" in l:
                    _prog(65.0)
                elif "[download] Destination:" in l:
                    _prog(73.0)
                elif dm := re.match(r"\[download\]\s+([\d.]+)%", l):
                    _prog(73.0 + float(dm.group(1)) * 0.09)
        proc.wait()

        try:
            vtt_check = pick_vtt(tmp_dir)
        except FileNotFoundError:
            if proc.returncode != 0:
                raise FileNotFoundError("Субтитры не найдены (.vtt отсутствует)")
            raise
        if proc.returncode != 0:
            _log("(yt-dlp завершился с ошибкой, но субтитры получены — продолжаем)\n")

        _log("Cleaning subtitles...\n")
        _prog(85.0)
        return clean_vtt(vtt_check)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_download(url: str, log_q: "queue.Queue") -> str:
    """
    Качает субтитры, чистит, сохраняет .txt в Загрузки.
    Пишет ход выполнения в log_q. Возвращает путь к готовому .txt.
    Бросает исключение при ошибке.
    """
    text = fetch_subtitle_text(url, log_q, progress_cb=None)

    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    name = f"subtitles_{m.group(1)}.txt" if m else "subtitles.txt"
    os.makedirs(DOWNLOADS, exist_ok=True)
    out_path = os.path.join(DOWNLOADS, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    log_q.put(f"Saved: {out_path}\n")
    log_q.put(100.0)
    return out_path


# форсируем UTF-8 в выводе yt-dlp, иначе кириллица в путях/логе бьётся в "ромбики"
def _utf8_env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _list_playlist_videos(url: str, env: dict, log_q: "queue.Queue") -> "list[str]":
    """Быстро (без скачивания) получает список id видео канала / плейлиста."""
    cmd = [
        "yt-dlp",
        *( ["--cookies-from-browser", BROWSER] if BROWSER else [] ),
        "--flat-playlist",
        "--print", "%(id)s",
        "--yes-playlist",
        url,
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        creationflags=NO_WINDOW, env=env,
    )
    ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not ids:
        raise RuntimeError("Не удалось получить список видео (проверьте ссылку и cookies)")
    return ids


def _download_one_video(video_url: str, langs: str, out_template: str,
                        env: dict, log_q: "queue.Queue") -> "list[str]":
    """Качает субтитры одного видео. Возвращает пути записанных .vtt."""
    cmd = [
        "yt-dlp",
        *( ["--cookies-from-browser", BROWSER] if BROWSER else [] ),
        "--remote-components", "ejs:github",
        "--write-auto-subs",
        "--skip-download",
        "--no-playlist",
        "--sub-langs", langs,
        # повторяем сбойные запросы (HTTP 429 и т.п.) с фиксированным кулдауном
        "--retries", str(SUB_RETRIES),
        "--retry-sleep", str(SUB_RETRY_SLEEP),
        "-o", out_template,
        "--newline",
        video_url,
    ]
    written: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=NO_WINDOW, env=env,
    )
    if proc.stdout is not None:
        for line in proc.stdout:
            log_q.put(line)
            wm = re.search(r"Writing video subtitles to:\s*(.+\.vtt)\s*$", line)
            if wm:
                written.append(wm.group(1).strip())
    proc.wait()
    return written


def run_download_playlist(url: str, langs: str, log_q: "queue.Queue") -> str:
    """
    Пакетно качает авто-субтитры со всего канала / плейлиста.
    langs - языки для скачивания (формат yt-dlp, например "ru" или "ru,en").
    Видео качаются в PLAYLIST_WORKERS параллельных потоков (1 = последовательно).
    Раскладывает файлы по подпапкам вида <канал>/<название> в Загрузках.
    Скачанные .vtt чистятся в .txt (как для одиночного видео), сами .vtt удаляются.
    Возвращает путь к папке Загрузок.
    """
    os.makedirs(DOWNLOADS, exist_ok=True)
    # шаблон вывода: Загрузки/<канал>/<название видео>.<ext>
    out_template = os.path.join(DOWNLOADS, "%(channel)s", "%(title)s.%(ext)s")
    env = _utf8_env()

    log_q.put("Fetching playlist items...\n")
    log_q.put(2.0)
    ids = _list_playlist_videos(url, env, log_q)
    total = len(ids)
    log_q.put(f"Found {total} video(s), downloading with {PLAYLIST_WORKERS} worker(s)...\n")
    log_q.put(5.0)

    # параллельно качаем субтитры по каждому видео
    written_vtts: list[str] = []
    workers = max(1, PLAYLIST_WORKERS)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _download_one_video,
                f"https://www.youtube.com/watch?v={vid}", langs, out_template, env, log_q,
            )
            for vid in ids
        ]
        for fut in as_completed(futures):
            try:
                written_vtts.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                log_q.put(f"Video failed: {e}\n")
            done += 1
            # прогресс: скачивание занимает диапазон 5..95%
            log_q.put(5.0 + (done / total) * 90.0)

    # чистим .vtt в .txt: по одному файлу на видео (язык — по LANG_PRIORITY)
    log_q.put("Cleaning subtitles...\n")
    log_q.put(95.0)
    # группируем .vtt одного видео, отрезая суффикс ".<язык>.vtt"
    groups: dict[str, list[str]] = {}
    for vtt in written_vtts:
        base = re.sub(r"\.[A-Za-z-]+\.vtt$", "", vtt)
        groups.setdefault(base, []).append(vtt)

    def _lang_of(path: str) -> str:
        lm = re.search(r"\.([A-Za-z-]+)\.vtt$", path)
        return lm.group(1) if lm else ""

    count = 0
    for base, vtts in groups.items():
        # выбираем язык по приоритету, иначе первый попавшийся
        chosen = next(
            (v for lang in LANG_PRIORITY for v in vtts if _lang_of(v) == lang),
            vtts[0],
        )
        try:
            text = clean_vtt(chosen)
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(text)
            count += 1
        except Exception as e:  # noqa: BLE001
            log_q.put(f"Skip (clean failed): {chosen} ({e})\n")
            continue
        # удаляем исходные .vtt этого видео (оставляем только .txt)
        for v in vtts:
            try:
                os.remove(v)
            except OSError:
                pass

    # ошибка только если совсем ничего не сохранили; частичные сбои (например
    # HTTP 429 по части видео) не считаем фатальными - что скачалось, то очистили
    if count == 0:
        raise RuntimeError("yt-dlp завершился с ошибкой, субтитры не получены")

    log_q.put(f"Done: {count} of {total} file(s) saved to {DOWNLOADS}\n")
    log_q.put(100.0)
    return DOWNLOADS


# ======================================================================
#  ДЕЙСТВИЯ С ГОТОВЫМ ФАЙЛОМ (Windows)
# ======================================================================

def open_folder(path: str):
    """Открыть Проводник и выделить файл."""
    subprocess.run(["explorer", "/select,", os.path.normpath(path)])


def open_file(path: str):
    """Открыть файл программой по умолчанию."""
    os.startfile(path)  # есть только на Windows


def copy_file_to_clipboard(path: str):
    """Скопировать сам файл в буфер (потом вставляется в Проводник / окно загрузки)."""
    env = os.environ.copy()
    env["_SUBSTRAY_CLIP"] = path
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Set-Clipboard -LiteralPath $env:_SUBSTRAY_CLIP"],
        creationflags=NO_WINDOW,
        env=env,
    )


def copy_text_to_clipboard(text: str):
    """Скопировать текст в буфер обмена."""
    env = os.environ.copy()
    env["_SUBSTRAY_TEXT"] = text
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $env:_SUBSTRAY_TEXT"],
        creationflags=NO_WINDOW,
        env=env,
    )


# ======================================================================
#  ИНТЕРФЕЙС (всё в главном потоке)
# ======================================================================

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)     # убираем из оконного менеджера Windows
        self.root.geometry("1x1+-200+-200")  # прячем за пределы экрана
        self.root.withdraw()
        self.cmd_q = queue.Queue()           # команды от иконки трея -> главный поток

        self.icon = self._build_icon()
        self.icon.run_detached()             # иконка крутится в фоновом потоке

        if os.name == "nt":
            threading.Thread(target=self._hotkey_listener, daemon=True).start()

        self.root.after(100, self._poll_commands)

    # ---- иконка трея ----
    def _make_image(self, progress: float | None = None):
        """Иконка с тремя полосками субтитров. progress=0..100 рисует дугу прогресса по кругу."""
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([6, 6, 58, 58], radius=10, fill=(45, 48, 56))
        for y in (24, 34, 44):
            d.rounded_rectangle([16, y, 48, y + 5], radius=2, fill=(120, 200, 160))
        if progress is not None:
            # тонкое серое кольцо-фон
            d.arc([4, 4, 60, 60], start=0, end=360, fill=(70, 73, 80), width=4)
            # зелёная дуга прогресса (от 12 часов по часовой стрелке)
            end_angle = -90 + (progress / 100) * 360
            if progress > 0:
                d.arc([4, 4, 60, 60], start=-90, end=end_angle, fill=(120, 200, 160), width=4)
        return img

    def _build_icon(self):
        menu = pystray.Menu(
            pystray.MenuItem("Скачать субтитры", self._on_download, default=True),
            pystray.MenuItem("Скачать с канала / плейлиста", self._on_download_playlist),
            pystray.MenuItem("Выход", self._on_quit),
        )
        return pystray.Icon("SubsToText", self._make_image(), "SubsToText", menu)

    # ---- глобальная горячая клавиша (Ctrl+Alt+Y) ----
    def _hotkey_listener(self):
        """Регистрирует горячую клавишу и ждёт WM_HOTKEY в собственном цикле сообщений."""
        ok = ctypes.windll.user32.RegisterHotKey(None, HOTKEY_ID, HOTKEY_MODS, HOTKEY_VK)
        if not ok:
            print("WARNING: не удалось зарегистрировать горячую клавишу Ctrl+Alt+Y")
            return
        msg = wintypes.MSG()
        while ctypes.windll.user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == 0x0312:  # WM_HOTKEY
                self.cmd_q.put("download_auto")
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
        ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_ID)

    # колбэки трея работают в потоке иконки -> просто кладём команду в очередь
    def _on_download(self, icon, item):
        self.cmd_q.put("download")

    def _on_download_playlist(self, icon, item):
        self.cmd_q.put("download_playlist")

    def _on_quit(self, icon, item):
        self.cmd_q.put("quit")

    # ---- диспетчер команд (главный поток) ----
    def _poll_commands(self):
        try:
            while True:
                cmd = self.cmd_q.get_nowait()
                if cmd == "download":
                    self._start_download_flow()
                elif cmd == "download_auto":
                    self._start_download_auto()
                elif cmd == "download_playlist":
                    self._start_playlist_flow()
                elif cmd == "quit":
                    self.icon.stop()
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_commands)

    # ---- окно ввода ссылки ----
    def _ask_url(self) -> str | None:
        win = tk.Toplevel(self.root)
        win.title("SubsToText - ссылка на видео")
        win.geometry("460x130")
        win.attributes("-topmost", True)
        win.grab_set()

        tk.Label(win, text="Вставьте ссылку на YouTube-видео:").pack(pady=(14, 4))
        var = tk.StringVar()

        # автоподстановка ссылки из буфера, если она youtube-овская
        try:
            clip = self.root.clipboard_get()
            if "youtu" in clip:
                var.set(clip.strip())
        except tk.TclError:
            pass

        entry = tk.Entry(win, textvariable=var, width=58)
        entry.pack(padx=14)
        entry.focus_set()
        entry.icursor(tk.END)

        def _paste(e=None):
            try:
                entry.delete(0, "end")
                entry.insert(0, win.clipboard_get())
            except tk.TclError:
                pass
            return "break"

        entry.bind("<<Paste>>", _paste)
        def _select_all(e=None):
            entry.select_range(0, "end")
            entry.icursor("end")
            return "break"

        # keycode 86 = V, 65 = A — независимо от раскладки (ru/en)
        entry.bind("<<SelectAll>>", _select_all)
        entry.bind("<Control-KeyPress>", lambda e: _paste() if e.keycode == 86 else _select_all() if e.keycode == 65 else None)

        result: dict[str, str | None] = {"url": None}

        def ok():
            result["url"] = var.get().strip()
            win.destroy()

        def cancel():
            win.destroy()

        btns = tk.Frame(win)
        btns.pack(pady=12)
        tk.Button(btns, text="Скачать", width=12, command=ok).pack(side="left", padx=6)
        tk.Button(btns, text="Отмена", width=12, command=cancel).pack(side="left", padx=6)
        win.bind("<Return>", lambda e: ok())
        win.bind("<Escape>", lambda e: cancel())

        self.root.wait_window(win)   # ждём закрытия окна
        return result["url"] or None

    # ---- окно ввода ссылки на канал / плейлист ----
    def _ask_url_playlist(self) -> "tuple[str, str] | None":
        win = tk.Toplevel(self.root)
        win.title("Канал / плейлист")
        win.geometry("500x180")
        win.attributes("-topmost", True)
        win.grab_set()

        tk.Label(win, text="Вставьте ссылку на канал или плейлист YouTube:").pack(pady=(14, 4))
        var = tk.StringVar()

        # автоподстановка ссылки из буфера, если она youtube-овская
        try:
            clip = self.root.clipboard_get()
            if "youtu" in clip:
                var.set(clip.strip())
        except tk.TclError:
            pass

        entry = tk.Entry(win, textvariable=var, width=64)
        entry.pack(padx=14)
        entry.focus_set()
        entry.icursor(tk.END)

        def _paste(e=None):
            try:
                entry.delete(0, "end")
                entry.insert(0, win.clipboard_get())
            except tk.TclError:
                pass
            return "break"

        entry.bind("<<Paste>>", _paste)
        def _select_all(e=None):
            entry.select_range(0, "end")
            entry.icursor("end")
            return "break"

        # keycode 86 = V, 65 = A — независимо от раскладки (ru/en)
        entry.bind("<<SelectAll>>", _select_all)
        entry.bind("<Control-KeyPress>", lambda e: _paste() if e.keycode == 86 else _select_all() if e.keycode == 65 else None)

        # выбор языка субтитров: качаем только один, как выбрали
        avail = [code.strip() for code in LANGS.split(",") if code.strip()]
        lang_var = tk.StringVar(value=LANG_PRIORITY[0] if LANG_PRIORITY[0] in avail else avail[0])
        frame_lang = tk.Frame(win)
        frame_lang.pack(pady=(12, 0))
        tk.Label(frame_lang, text="Язык субтитров:").pack(side="left", padx=(0, 6))
        for code in avail:
            label = LANG_NAMES.get(code, code)
            tk.Radiobutton(frame_lang, text=label, variable=lang_var, value=code).pack(side="left", padx=3)

        result: dict[str, str | None] = {"url": None, "lang": None}

        def ok():
            result["url"] = var.get().strip()
            result["lang"] = lang_var.get()
            win.destroy()

        def cancel():
            win.destroy()

        btns = tk.Frame(win)
        btns.pack(pady=12)
        tk.Button(btns, text="Скачать всё", width=12, command=ok).pack(side="left", padx=6)
        tk.Button(btns, text="Отмена", width=12, command=cancel).pack(side="left", padx=6)
        win.bind("<Return>", lambda e: ok())
        win.bind("<Escape>", lambda e: cancel())

        self.root.wait_window(win)   # ждём закрытия окна
        if not result["url"]:
            return None
        return result["url"], result["lang"] or avail[0]

    # ---- общий каркас: окно прогресса + фоновый воркер ----
    def _run_with_progress(self, title: str, worker_fn, on_done):
        prog = tk.Toplevel(self.root)
        prog.title(title)
        prog.geometry("560x300")
        prog.attributes("-topmost", True)
        prog.grab_set()

        bar = ttk.Progressbar(prog, mode="determinate", maximum=100)
        bar.pack(fill="x", padx=12, pady=(12, 6))

        log = tk.Text(prog, height=12, wrap="word", font=("Consolas", 9))
        log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        log_q: queue.Queue = queue.Queue()
        result: dict[str, object] = {"path": None, "error": None, "done": False}

        threading.Thread(target=worker_fn, args=(log_q, result), daemon=True).start()

        def pump():
            try:
                while True:
                    msg = log_q.get_nowait()
                    if isinstance(msg, float):
                        bar["value"] = msg
                    else:
                        log.insert("end", msg)
                        log.see("end")
            except queue.Empty:
                pass

            if result["done"]:
                if result.get("error"):
                    prog.grab_release()
                    tk.Button(prog, text="Закрыть", command=prog.destroy).pack(pady=4)
                else:
                    bar["value"] = 100
                    prog.destroy()
                on_done(result)
                return
            prog.after(120, pump)

        prog.after(120, pump)

    # ---- быстрый запуск по горячей клавише: URL берётся из буфера ----
    def _start_download_auto(self):
        try:
            url = self.root.clipboard_get().strip()
        except tk.TclError:
            url = ""
        if not url or "youtu" not in url:
            self.icon.notify("В буфере обмена нет YouTube-ссылки", "SubsToText")
            return
        self._run_auto_silent(url)

    def _run_auto_silent(self, url: str):
        """Тихое скачивание без окна прогресса — прогресс на иконке трея, результат в буфер."""
        def on_progress(pct: float):
            self.icon.icon = self._make_image(progress=pct)
            self.icon.title = f"SubsToText — {pct:.0f}%"

        def reset_icon():
            self.icon.icon = self._make_image()
            self.icon.title = "SubsToText"

        def worker():
            try:
                text = fetch_subtitle_text(url, log_q=None, progress_cb=on_progress)
                copy_text_to_clipboard(text)
                words = len(text.split())
                reset_icon()
                self.icon.notify(f"Субтитры скопированы (~{words} слов)", "SubsToText")
            except FileNotFoundError:
                reset_icon()
                self.icon.notify("Субтитры не найдены", "SubsToText")
            except Exception as e:  # noqa: BLE001
                reset_icon()
                self.icon.notify(f"Ошибка: {e}", "SubsToText")

        threading.Thread(target=worker, daemon=True).start()

    # ---- запуск процесса скачивания ----
    def _start_download_flow(self):
        url = self._ask_url()
        if not url:
            return
        self._run_download_url(url)

    def _run_download_url(self, url: str):
        def worker(log_q, result):
            try:
                result["path"] = run_download(url, log_q)
            except FileNotFoundError:
                result["no_subs"] = True
            except Exception as e:  # noqa: BLE001
                result["error"] = str(e)
            finally:
                result["done"] = True

        def on_done(result):
            if result.get("no_subs"):
                self._offer_whisper(url)
            elif result["error"]:
                messagebox.showerror(
                    "Ошибка",
                    "Не удалось скачать субтитры:\n\n"
                    + str(result["error"])
                    + "\n\nПроверьте ссылку, cookies (Firefox) и доступность видео.",
                )
            else:
                self._show_result(str(result["path"]))

        self._run_with_progress("SubsToText - скачивание...", worker, on_done)

    # ---- запуск пакетного скачивания канала / плейлиста ----
    def _start_playlist_flow(self):
        answer = self._ask_url_playlist()
        if not answer:
            return
        url, lang = answer
        self._run_download_playlist(url, lang)

    def _run_download_playlist(self, url: str, lang: str):
        def worker(log_q, result):
            try:
                result["path"] = run_download_playlist(url, lang, log_q)
            except Exception as e:  # noqa: BLE001
                result["error"] = str(e)
            finally:
                result["done"] = True

        def on_done(result):
            if result["error"]:
                messagebox.showerror(
                    "Ошибка",
                    "Не удалось скачать субтитры с канала / плейлиста:\n\n"
                    + str(result["error"])
                    + "\n\nПроверьте ссылку, cookies (Firefox) и доступность видео.",
                )
            else:
                # открываем папку Загрузок в Проводнике
                subprocess.run(["explorer", os.path.normpath(str(result["path"]))])

        self._run_with_progress("SubsToText - скачивание плейлиста...", worker, on_done)

    # ---- окно результата: что делать с готовым файлом ----
    def _show_result(self, path: str):
        win = tk.Toplevel(self.root)
        win.title("Готово")
        win.geometry("420x170")
        win.attributes("-topmost", True)
        win.grab_set()
        win.bind("<Escape>", lambda _: win.destroy())

        tk.Label(win, text="Субтитры сохранены:", font=("", 10, "bold")).pack(pady=(14, 2))
        tk.Label(win, text=os.path.basename(path), fg="#2a7").pack()

        def folder():
            open_folder(path); win.destroy()

        def openf():
            open_file(path); win.destroy()

        def clip():
            copy_file_to_clipboard(path); win.destroy()

        frame = tk.Frame(win)
        frame.pack(pady=18)
        tk.Button(frame, text="Открыть папку", width=15, command=folder).pack(side="left", padx=4)
        tk.Button(frame, text="Открыть файл", width=15, command=openf).pack(side="left", padx=4)
        tk.Button(frame, text="Копировать файл", width=15, command=clip).pack(side="left", padx=4)

    # ---- предложение транскрибировать через Whisper ----
    def _offer_whisper(self, url: str):
        win = tk.Toplevel(self.root)
        win.title("Субтитры не найдены")
        win.geometry("420x200")
        win.attributes("-topmost", True)
        win.grab_set()

        tk.Label(win, text="У этого видео нет субтитров.", font=("", 10, "bold")).pack(pady=(16, 6))

        if not whisper_available():
            tk.Label(win, text="Можно транскрибировать аудио офлайн через Whisper.\nДля этого установите пакет:").pack()
            tk.Label(win, text="pip install faster-whisper", font=("Consolas", 9), fg="#555").pack(pady=6)
            tk.Button(win, text="Закрыть", width=12, command=win.destroy).pack(pady=8)
            return

        tk.Label(win, text="Транскрибировать аудио через Whisper?").pack()

        frame_model = tk.Frame(win)
        frame_model.pack(pady=10)
        tk.Label(frame_model, text="Модель:").pack(side="left", padx=(0, 6))
        model_var = tk.StringVar(value="base")
        for m in ("tiny", "base", "small", "medium"):
            tk.Radiobutton(frame_model, text=m, variable=model_var, value=m).pack(side="left", padx=3)

        def transcribe():
            win.destroy()
            self._start_whisper_flow(url, model_var.get())

        btns = tk.Frame(win)
        btns.pack(pady=8)
        tk.Button(btns, text="Транскрибировать", width=18, command=transcribe).pack(side="left", padx=6)
        tk.Button(btns, text="Отмена", width=12, command=win.destroy).pack(side="left", padx=6)

    # ---- запуск транскрипции ----
    def _start_whisper_flow(self, url: str, model_size: str):
        def worker(log_q, result):
            try:
                result["path"] = run_whisper(url, model_size, log_q)
            except Exception as e:  # noqa: BLE001
                result["error"] = str(e)
            finally:
                result["done"] = True

        def on_done(result):
            if result["error"]:
                messagebox.showerror(
                    "Ошибка транскрипции",
                    "Не удалось транскрибировать:\n\n" + str(result["error"]),
                )
            else:
                self._show_result(str(result["path"]))

        self._run_with_progress("SubsToText - транскрипция...", worker, on_done)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
