import cv2
import threading
import time

class VideoStream:
    """
    Class untuk menangani pengambilan frame dari kamera menggunakan thread terpisah
    agar tidak menghambat (blocking) proses utama aplikasi.
    """
    def __init__(self, src=0, width=640, height=480):
        # Inisialisasi kamera
        self.cap = cv2.VideoCapture(src)
        
        # Atur resolusi (Default 640x480 untuk efisiensi laptop/Jetson)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Ambil frame pertama
        self.ret, self.frame = self.cap.read()
        
        # Status thread
        self.stopped = False

    def start(self):
        """Memulai thread untuk membaca frame secara background."""
        # daemon=True memastikan thread berhenti saat program utama dimatikan
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            if not self.cap.isOpened():
                self.stop()
            time.sleep(0.01) # Sleep sebentar untuk mengurangi beban CPU

    def read(self):
        """Mengembalikan frame terbaru yang berhasil ditangkap."""
        # Mengembalikan salinan (copy) agar frame asli tidak terganggu saat diolah AI
        return self.frame.copy() if self.ret else None

    def stop(self):
        """Menghentikan thread dan melepaskan resource kamera."""
        self.stopped = True
        self.cap.release()