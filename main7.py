import numpy as np
import cv2
import sys
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESH_RED_PEAK = 300.0  # ✅ bandpass*alpha 紅線最大值未達此門檻 => BPM=0

# ---------------- Helper Methods ----------------
def buildGauss(frame, levels):
    pyramid = [frame]
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels, out_h, out_w):
    filteredFrame = pyramid[index]
    for _ in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    return filteredFrame[:out_h, :out_w]

def put_info(img, text, org=(20, 30), color=(0, 255, 0), scale=0.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def bgr_to_tk(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

def rgb_to_tk(img_rgb):
    return ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape((h, w, 3)).copy()
    plt.close(fig)
    return img

def plot_fft_to_rgb(freqs, amp_green, amp_red, width=450, height=220, title="FFT Spectrum"):
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    plotted_any = False
    if freqs is not None and amp_green is not None and len(freqs) == len(amp_green) and len(freqs) > 0:
        ax.plot(freqs, amp_green, 'g-', linewidth=2, label="raw FFT (green)")
        plotted_any = True
    if freqs is not None and amp_red is not None and len(freqs) == len(amp_red) and len(freqs) > 0:
        ax.plot(freqs, amp_red, 'r-', linewidth=2, label="bandpass*alpha (red)")
        plotted_any = True

    if plotted_any:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig_to_rgb(fig)

# ---------------- App ----------------
class MotionMagApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cam4 | Motion Mag (ROI1+ROI2) + ROI FFTs + ROI BPMs (thresholded)")

        # -------- Shared state --------
        self.running = True
        self.processing = True
        self.lock = threading.Lock()

        self.last_orig = None
        self.last_mag = None

        # FFT plot data for ROI1/ROI2
        self.fft1_freqs = None
        self.fft1_g = None
        self.fft1_r = None
        self.fft2_freqs = None
        self.fft2_g = None
        self.fft2_r = None

        # BPM values
        self.bpm1 = 0.0
        self.bpm2 = 0.0

        # Metrics
        self.fps_ema = 0.0
        self.lat_ms_ema = 0.0

        # -------- UI variables --------
        self.display_scale_var = tk.DoubleVar(value=1.0)
        self.alpha_var = tk.IntVar(value=170)

        # -------- Layout: 2x2 grid --------
        grid = tk.Frame(root)
        grid.pack(fill="both", expand=True)
        for c in (0, 1):
            grid.columnconfigure(c, weight=1)
        grid.rowconfigure(0, weight=3)
        grid.rowconfigure(1, weight=1)

        self.lbl_orig = tk.Label(grid, bd=1, relief="sunken")
        self.lbl_orig.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.lbl_mag = tk.Label(grid, bd=1, relief="sunken")
        self.lbl_mag.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        self.lbl_fft1 = tk.Label(grid, bd=1, relief="sunken")
        self.lbl_fft1.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        self.lbl_fft2 = tk.Label(grid, bd=1, relief="sunken")
        self.lbl_fft2.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)

        # -------- Controls (無錄影按鍵) --------
        panel = tk.Frame(root)
        panel.pack(fill="x")

        row1 = tk.Frame(panel); row1.pack(fill="x", pady=2)
        self.btn_toggle = tk.Button(row1, text="Pause", width=10, command=self.toggle_processing)
        self.btn_toggle.pack(side="left", padx=6)
        tk.Button(row1, text="Quit", width=10, command=self.on_close).pack(side="right", padx=6)

        row2 = tk.Frame(panel); row2.pack(fill="x", pady=2)
        tk.Label(row2, text="Display Scale").pack(side="left", padx=6)
        tk.Scale(row2, from_=0.3, to=2.0, resolution=0.05, orient="horizontal",
                 variable=self.display_scale_var, length=320).pack(side="left", padx=6)
        tk.Label(row2, text="Alpha").pack(side="left", padx=6)
        tk.Scale(row2, from_=1, to=400, resolution=1, orient="horizontal",
                 variable=self.alpha_var, length=260).pack(side="left", padx=6)

        # Worker
        self.worker = threading.Thread(target=self.capture_loop, daemon=True)
        self.worker.start()

        # UI update
        self.update_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle_processing(self):
        with self.lock:
            self.processing = not self.processing
            self.btn_toggle.configure(text="Pause" if self.processing else "Start")

    def on_close(self):
        with self.lock:
            self.running = False
        time.sleep(0.05)
        try:
            self.root.destroy()
        except:
            pass

    def capture_loop(self):
        cap = cv2.VideoCapture(4, cv2.CAP_V4L2) if len(sys.argv) == 2 else cv2.VideoCapture(4)
        realW, realH = 640, 480
        videoFrameRate = 15
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, realW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, realH)
        cap.set(cv2.CAP_PROP_FPS, videoFrameRate)

        # Fixed ROIs
        roi1 = (30, 130, 180, 120)  # x,y,w,h
        roi2 = (380, 130, 180, 120)

        def clip_roi(x, y, w, h):
            x = max(0, min(realW - 2, x))
            y = max(0, min(realH - 2, y))
            w = max(2, min(realW - x, w))
            h = max(2, min(realH - y, h))
            return x, y, w, h

        x1, y1, w1, h1 = clip_roi(*roi1)
        x2, y2, w2, h2 = clip_roi(*roi2)

        levels = 3
        minFrequency = 1.0
        maxFrequency = 2.0
        bufferSize = 150
        bufferIndex = 0

        frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
        mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

        # independent buffers
        firstFrame1 = np.zeros((h1, w1, 3), dtype=np.uint8)
        firstGauss1 = buildGauss(firstFrame1, levels + 1)[levels]
        videoGauss1 = np.zeros((bufferSize, firstGauss1.shape[0], firstGauss1.shape[1], 3), dtype=np.float32)

        firstFrame2 = np.zeros((h2, w2, 3), dtype=np.uint8)
        firstGauss2 = buildGauss(firstFrame2, levels + 1)[levels]
        videoGauss2 = np.zeros((bufferSize, firstGauss2.shape[0], firstGauss2.shape[1], 3), dtype=np.float32)

        s1 = np.zeros((bufferSize,), dtype=np.float32)
        s2 = np.zeros((bufferSize,), dtype=np.float32)

        boxColor1 = (0, 255, 0)
        boxColor2 = (255, 0, 0)
        boxWeight = 2

        prev_t = time.perf_counter()
        update_every = 5

        while True:
            with self.lock:
                if not self.running:
                    break
                if not self.processing:
                    time.sleep(0.01)
                    continue
                alpha = float(self.alpha_var.get())

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            roi1_img = frame[y1:y1+h1, x1:x1+w1, :]
            roi2_img = frame[y2:y2+h2, x2:x2+w2, :]

            gauss1 = buildGauss(roi1_img, levels + 1)[levels].astype(np.float32)
            gauss2 = buildGauss(roi2_img, levels + 1)[levels].astype(np.float32)

            videoGauss1[bufferIndex] = gauss1
            videoGauss2[bufferIndex] = gauss2
            s1[bufferIndex] = float(gauss1.mean())
            s2[bufferIndex] = float(gauss2.mean())

            # motion mag ROI1/ROI2
            ft1 = np.fft.fft(videoGauss1, axis=0)
            ft2 = np.fft.fft(videoGauss2, axis=0)
            ft1_bp = ft1.copy(); ft1_bp[~mask] = 0
            ft2_bp = ft2.copy(); ft2_bp[~mask] = 0

            filtered1 = np.real(np.fft.ifft(ft1_bp, axis=0)).astype(np.float32) * alpha
            filtered2 = np.real(np.fft.ifft(ft2_bp, axis=0)).astype(np.float32) * alpha

            filteredFrame1 = reconstructFrame(filtered1, bufferIndex, levels, h1, w1)
            filteredFrame2 = reconstructFrame(filtered2, bufferIndex, levels, h2, w2)

            out_roi1 = cv2.convertScaleAbs(roi1_img.astype(np.float32) + filteredFrame1)
            out_roi2 = cv2.convertScaleAbs(roi2_img.astype(np.float32) + filteredFrame2)

            bufferIndex = (bufferIndex + 1) % bufferSize

            orig_view = frame.copy()
            mag_view = frame.copy()
            mag_view[y1:y1+h1, x1:x1+w1, :] = out_roi1
            mag_view[y2:y2+h2, x2:x2+w2, :] = out_roi2

            cv2.rectangle(orig_view, (x1, y1), (x1+w1, y1+h1), boxColor1, boxWeight)
            cv2.rectangle(orig_view, (x2, y2), (x2+w2, y2+h2), boxColor2, boxWeight)
            cv2.rectangle(mag_view,  (x1, y1), (x1+w1, y1+h1), boxColor1, boxWeight)
            cv2.rectangle(mag_view,  (x2, y2), (x2+w2, y2+h2), boxColor2, boxWeight)

            # FPS/Latency
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            inst_fps = 1.0 / dt if dt > 1e-6 else 0.0
            inst_lat_ms = dt * 1000.0
            self.fps_ema = inst_fps if self.fps_ema == 0 else (0.9 * self.fps_ema + 0.1 * inst_fps)
            self.lat_ms_ema = inst_lat_ms if self.lat_ms_ema == 0 else (0.9 * self.lat_ms_ema + 0.1 * inst_lat_ms)

            # ---- Update FFT lines + BPM with threshold rule ----
            if (bufferIndex % update_every) == 0:
                S1 = np.fft.fft(s1)
                S2 = np.fft.fft(s2)

                # green lines
                amp1_g = np.abs(S1).astype(np.float32)
                amp2_g = np.abs(S2).astype(np.float32)

                # red lines = bandpass * alpha
                S1_bp = S1.copy(); S1_bp[~mask] = 0
                S2_bp = S2.copy(); S2_bp[~mask] = 0
                amp1_r = (np.abs(S1_bp) * alpha).astype(np.float32)
                amp2_r = (np.abs(S2_bp) * alpha).astype(np.float32)

                # threshold decision (use red peak)
                red_peak1 = float(np.max(amp1_r)) if amp1_r.size else 0.0
                red_peak2 = float(np.max(amp2_r)) if amp2_r.size else 0.0

                if red_peak1 < THRESH_RED_PEAK:
                    bpm1 = 0.0
                else:
                    idx1 = int(np.argmax(np.abs(S1_bp)))
                    bpm1 = float(60.0 * frequencies[idx1])

                if red_peak2 < THRESH_RED_PEAK:
                    bpm2 = 0.0
                else:
                    idx2 = int(np.argmax(np.abs(S2_bp)))
                    bpm2 = float(60.0 * frequencies[idx2])

                with self.lock:
                    self.fft1_freqs = frequencies.copy()
                    self.fft1_g = amp1_g
                    self.fft1_r = amp1_r
                    self.fft2_freqs = frequencies.copy()
                    self.fft2_g = amp2_g
                    self.fft2_r = amp2_r
                    self.bpm1 = bpm1
                    self.bpm2 = bpm2

            with self.lock:
                bpm1_disp = float(self.bpm1)
                bpm2_disp = float(self.bpm2)

            # overlay texts (BPM forced to 0 by threshold)
            put_info(orig_view, f"Cam4 Input", (20, 30))
            put_info(mag_view,  f"Motion Mag Output (ROI1+ROI2)",org=(420, 30), scale=0.60)
            put_info(mag_view,  f"alpha={int(alpha)}", org=(420, 65), scale=0.60)
            #put_info(mag_view,  f"ROI1=Green box, ROI2=Blue box", (5, 65), scale=0.7)

            put_info(mag_view, f"ROI1 BPM: {int(bpm1_disp)}", (20, 125), scale=0.6)
            put_info(mag_view, f"ROI2 BPM: {int(bpm2_disp)}", (370, 125), scale=0.6,color=(255,0,0))

            put_info(mag_view, f"FPS: {self.fps_ema:.1f}", org=(5, 35), scale=0.70)
            put_info(mag_view, f"Latency: {self.lat_ms_ema:.1f} ms", org=(420, 90), scale=0.60 )

            with self.lock:
                self.last_orig = orig_view
                self.last_mag = mag_view

        cap.release()

    def update_ui(self):
        with self.lock:
            if not self.running:
                return
            orig = None if self.last_orig is None else self.last_orig.copy()
            mag  = None if self.last_mag  is None else self.last_mag.copy()
            processing = self.processing

            f1 = None if self.fft1_freqs is None else self.fft1_freqs.copy()
            g1 = None if self.fft1_g is None else self.fft1_g.copy()
            r1 = None if self.fft1_r is None else self.fft1_r.copy()

            f2 = None if self.fft2_freqs is None else self.fft2_freqs.copy()
            g2 = None if self.fft2_g is None else self.fft2_g.copy()
            r2 = None if self.fft2_r is None else self.fft2_r.copy()

        scale = float(self.display_scale_var.get())

        if orig is not None:
            if scale != 1.0:
                orig = cv2.resize(orig, (int(orig.shape[1]*scale), int(orig.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
            if not processing:
                put_info(orig, "PAUSED", org=(20, orig.shape[0]-20), color=(0, 255, 255))
            tk_img = bgr_to_tk(orig)
            self.lbl_orig.imgtk = tk_img
            self.lbl_orig.configure(image=tk_img)

        if mag is not None:
            if scale != 1.0:
                mag = cv2.resize(mag, (int(mag.shape[1]*scale), int(mag.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
            if not processing:
                put_info(mag, "PAUSED", org=(20, mag.shape[0]-20), color=(0, 255, 255))
            tk_img = bgr_to_tk(mag)
            self.lbl_mag.imgtk = tk_img
            self.lbl_mag.configure(image=tk_img)

        fft1_rgb = plot_fft_to_rgb(f1, g1, r1, width=450, height=220,
                                   title=f"ROI1 FFT (red_peak<{THRESH_RED_PEAK:.0f} => BPM=0)")
        tk_fft1 = rgb_to_tk(fft1_rgb)
        self.lbl_fft1.imgtk = tk_fft1
        self.lbl_fft1.configure(image=tk_fft1)

        fft2_rgb = plot_fft_to_rgb(f2, g2, r2, width=450, height=220,
                                   title=f"ROI2 FFT (red_peak<{THRESH_RED_PEAK:.0f} => BPM=0)")
        tk_fft2 = rgb_to_tk(fft2_rgb)
        self.lbl_fft2.imgtk = tk_fft2
        self.lbl_fft2.configure(image=tk_fft2)

        self.root.after(60, self.update_ui)


if __name__ == "__main__":
    root = tk.Tk()
    app = MotionMagApp(root)
    root.mainloop()

