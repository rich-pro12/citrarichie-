import cv2
import numpy as np
import matplotlib.pyplot as plt

# Memuat Pengklasifikasi Kaskade Haar (Pastikan file XML ada di folder yang sama atau sesuaikan path-nya)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Membuat Fungsi untuk Mendeteksi Wajah
def adjusted_detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 255, 255), 10)
    return face_img

# Membuat Fungsi untuk Mendeteksi Mata
def detect_eyes(img):
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in eye_rect:
        cv2.rectangle(eye_img, (x, y), (x+w, y+h), (255, 255, 255), 10)
    return eye_img

# Memuat Gambar
img = cv2.imread('C:/xampp/htdocs/Citra Richie/richie.jpeg') # <-- Ubah dengan nama file fotomu
img_copy1 = img.copy()
img_copy2 = img.copy()
img_copy3 = img.copy()

# Mendeteksi Wajah dan Mata
face = adjusted_detect_face(img_copy1)
eyes = detect_eyes(img_copy2)

# Menggabungkan deteksi wajah dan mata pada satu gambar
eyes_face = adjusted_detect_face(img_copy3)
eyes_face = detect_eyes(eyes_face)

# Menyimpan hasil gambar
cv2.imwrite('face.jpg', face)
cv2.imwrite('eyes.jpg', eyes)
cv2.imwrite('face+eyes.jpg', eyes_face)

print("Proses selesai! Silakan periksa file face.jpg, eyes.jpg, dan face+eyes.jpg di folder proyekmu.")