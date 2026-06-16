#!/usr/bin/env python3
"""
SubsToText - приложение в трее Windows.
Клик по иконке -> вставить ссылку на YouTube -> скачать авто-субтитры,
почистить от мусора и сохранить .txt в папку "Загрузки".
После успеха: открыть папку / открыть файл / скопировать файл в буфер.

Зависимости: pip install pystray pillow
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
LANGS = "ru,en"                   # языки субтитров
LANG_PRIORITY = ["ru", "en"]      # предпочтительный язык на выходе
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
# CREATE_NO_WINDOW - чтобы не мигали чёрные окна консоли при вызове yt-dlp
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

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
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=5)
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
