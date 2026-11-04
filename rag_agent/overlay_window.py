"""Windows-only, mouse-interactive overlay that does not activate on clicks."""

import ctypes
import json
import queue
import sys
import threading
import tkinter as tk
from ctypes import wintypes
from urllib.request import Request, urlopen


STATE_URL = "http://127.0.0.1:8000/api/live/overlay/state"
WS_EX_NOACTIVATE = 0x08000000
GWL_EXSTYLE = -20
GWL_WNDPROC = -4
WM_MOUSEACTIVATE = 0x0021
MA_NOACTIVATE = 3
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
GA_ROOT = 2


def install_no_activate(root: tk.Tk):
    """Keep the meeting window foreground while this overlay receives mouse clicks."""
    user32 = ctypes.windll.user32
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallWindowProcW.restype = ctypes.c_ssize_t
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.SetWindowPos.restype = wintypes.BOOL

    hwnd = user32.GetAncestor(root.winfo_id(), GA_ROOT)
    if not hwnd:
        raise RuntimeError("无法取得浮窗句柄")
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE)

    procedure = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    previous = ctypes.c_ssize_t()

    @procedure
    def window_proc(window, message, wparam, lparam):
        if message == WM_MOUSEACTIVATE:
            return MA_NOACTIVATE
        return user32.CallWindowProcW(ctypes.c_void_p(previous.value), window, message, wparam, lparam)

    previous.value = user32.SetWindowLongPtrW(hwnd, GWL_WNDPROC, ctypes.cast(window_proc, ctypes.c_void_p).value)
    if not previous.value:
        raise RuntimeError("无法设置非激活浮窗")
    user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED)
    return hwnd, previous.value, window_proc


class OverlayWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Atlas 通话辅助")
        self.root.geometry("430x560+900+110")
        self.root.minsize(320, 250)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#f4f7fb")
        self.header = tk.Canvas(self.root, bg="#21314f", height=48, highlightthickness=0, takefocus=0)
        self.header.pack(fill="x")
        self.header.bind("<ButtonPress-1>", self.start_drag)
        self.header.bind("<B1-Motion>", self.drag)
        self.header.bind("<ButtonRelease-1>", self.stop_drag)
        self.canvas = tk.Canvas(self.root, bg="#f4f7fb", highlightthickness=0, takefocus=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<MouseWheel>", self.scroll)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.grip = tk.Label(self.root, text="◢", fg="#8190a6", bg="#f4f7fb", cursor="sizing", takefocus=0)
        self.grip.place(relx=1, rely=1, anchor="se")
        self.grip.bind("<ButtonPress-1>", self.start_resize)
        self.grip.bind("<B1-Motion>", self.drag)
        self.grip.bind("<ButtonRelease-1>", self.stop_drag)
        self.root.update_idletasks()
        self.hwnd, self.previous_proc, self.window_proc = install_no_activate(self.root)
        self.user32 = ctypes.windll.user32
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.snapshot = {"running": False, "status": "连接本地服务中…", "entries": [],
                         "interviewer_preview": "", "me_preview": ""}
        self.updated_at = ""
        self.drag_state = None
        threading.Thread(target=self.poll, daemon=True).start()
        self.root.after(100, self.receive)
        self.draw()

    def poll(self):
        while not self.stop_event.is_set():
            try:
                request = Request(STATE_URL, headers={"X-Atlas-Overlay": "atlas-live"})
                with urlopen(request, timeout=3) as response:
                    data = json.load(response)
                self.events.put(data)
            except (OSError, ValueError):
                self.events.put({"error": "本地服务暂时无法连接"})
            self.stop_event.wait(0.35)

    def receive(self):
        latest = None
        while not self.events.empty():
            latest = self.events.get_nowait()
        if latest:
            if "snapshot" in latest:
                self.snapshot = latest["snapshot"]
                self.updated_at = latest.get("updated_at", "")
            else:
                self.snapshot["status"] = latest.get("error", "连接中…")
            self.draw()
        self.root.after(100, self.receive)

    def text(self, x, y, value, width, *, color="#24344d", size=11, weight="normal"):
        item = self.canvas.create_text(x, y, text=value, width=width, anchor="nw",
                                       fill=color, font=("Microsoft YaHei UI", size, weight))
        bounds = self.canvas.bbox(item)
        return (bounds[3] - bounds[1]) if bounds else 0

    def draw(self):
        canvas = self.canvas
        old_bottom = canvas.yview()[1] >= 0.96
        canvas.delete("all")
        width = max(self.root.winfo_width(), 320)
        height = max(canvas.winfo_height(), 202)
        self.header.delete("all")
        self.header.create_text(15, 13, text="ATLAS  通话辅助", anchor="nw", fill="white",
                                font=("Microsoft YaHei UI", 13, "bold"))
        self.header.create_text(width - 30, 8, text="×", anchor="nw", fill="white",
                                font=("Microsoft YaHei UI", 17))
        state = "录制中" if self.snapshot.get("running") else "待命"
        canvas.create_rectangle(10, 10, width - 10, 64, fill="white", outline="#e1e8f2")
        self.text(20, 17, state, width - 40, color="#2d68dc", size=10, weight="bold")
        self.text(20, 36, self.snapshot.get("status", "")[:160], width - 40, color="#667892", size=9)
        y = 76
        for role, title in (("interviewer", "面试官 · 电脑声音"), ("me", "我 · 麦克风")):
            preview = self.snapshot.get(f"{role}_preview", "") or "等待语音…"
            canvas.create_rectangle(10, y, width - 10, y + 58, fill="#f9fbff", outline="#dfe7f2")
            self.text(20, y + 6, title, width - 40, color="#60728d", size=9, weight="bold")
            self.text(20, y + 28, preview[:100], width - 40, color="#354861", size=10)
            y += 68
        y += 6
        entries = self.snapshot.get("entries", [])[-8:]
        if not entries:
            self.text(18, y, "会话记录会显示在这里。", width - 36, color="#8190a6", size=10)
            y += 36
        for entry in entries:
            top = y
            role = "面试官" if entry.get("role") == "interviewer" else "我"
            color = "#2a9870" if role == "面试官" else "#4d76d3"
            label_height = self.text(22, y + 10, role, width - 44, color=color, size=10, weight="bold")
            y += label_height + 18
            y += self.text(22, y, entry.get("text", "")[:1500], width - 44) + 10
            if entry.get("answer") is not None:
                y += self.text(22, y, "AI 回答", width - 44, color="#315cbd", size=10, weight="bold") + 5
                y += self.text(22, y, (entry.get("answer") or "正在生成…")[:4000], width - 44, color="#31507e") + 9
                if entry.get("answer_tag"):
                    y += self.text(22, y, entry["answer_tag"], width - 44, color="#9a5a00", size=9) + 6
            if entry.get("assessment"):
                y += self.text(22, y, "我的回答评价", width - 44, color="#21694b", size=10, weight="bold") + 5
                y += self.text(22, y, entry["assessment"][:3000], width - 44, color="#315d4a") + 8
            y += 12
            rectangle = canvas.create_rectangle(10, top, width - 10, y, fill="white", outline="#dfe7f2")
            canvas.tag_lower(rectangle)
            canvas.create_rectangle(10, top, 13, y, fill=color, outline="")
            y += 10
        canvas.configure(scrollregion=(0, 0, width, max(y + 10, height)))
        if old_bottom:
            canvas.yview_moveto(1.0)

    def start_drag(self, event):
        width = self.root.winfo_width()
        if event.x > width - 48:
            self.close()
        else:
            self.drag_state = ("move", event.x_root - self.root.winfo_x(),
                               event.y_root - self.root.winfo_y())

    def start_resize(self, event):
        self.drag_state = ("resize", event.x_root, event.y_root,
                           self.root.winfo_width(), self.root.winfo_height())

    def drag(self, event):
        if not self.drag_state:
            return
        if self.drag_state[0] == "move":
            _, offset_x, offset_y = self.drag_state
            self.root.geometry(f"+{event.x_root - offset_x}+{event.y_root - offset_y}")
        else:
            _, start_x, start_y, width, height = self.drag_state
            new_width = max(320, width + event.x_root - start_x)
            new_height = max(250, height + event.y_root - start_y)
            self.root.geometry(f"{new_width}x{new_height}")

    def stop_drag(self, _):
        self.drag_state = None

    def scroll(self, event):
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def close(self):
        self.stop_event.set()
        self.user32.SetWindowLongPtrW(self.hwnd, GWL_WNDPROC, self.previous_proc)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    if sys.platform != "win32":
        raise SystemExit("原生浮窗目前仅支持 Windows")
    window = OverlayWindow()
    if "--self-test" in sys.argv:
        style = window.user32.GetWindowLongPtrW(window.hwnd, GWL_EXSTYLE)
        result = window.user32.SendMessageW(window.hwnd, WM_MOUSEACTIVATE, 0, 0)
        print(json.dumps({"no_activate_style": bool(style & WS_EX_NOACTIVATE),
                          "mouse_click_keeps_focus": result == MA_NOACTIVATE}))
        window.close()
    else:
        window.run()


if __name__ == "__main__":
    main()
