import cv2
import numpy as np

# Menyiapkan Pengambilan Video dari Webcam (0 adalah kamera default)
video = cv2.VideoCapture(0)

# Memuat Haar Cascade menggunakan path bawaan OpenCV (Mencegah error file tidak ditemukan)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Menginisialisasi Variabel
faces_data = [] # List untuk menyimpan data wajah yang dipotong
i = 0           # Penghitung frame

print("Memulai kamera... Tekan 'q' pada keyboard untuk menghentikan program.")

# Menangkap dan Memproses Bingkai secara Real-Time
while True:
    ret, frame = video.read()
    
    # Jika kamera gagal terbuka/terbaca, lewati loop
    if not ret:
        print("Gagal mengambil gambar dari kamera. Pastikan webcam tidak sedang digunakan oleh aplikasi lain.")
        break

    # Mengubah gambar menjadi hitam putih (Grayscale) untuk mempermudah deteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Mendeteksi wajah pada frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Memotong bagian wajah dari frame aslinya
        crop_img = frame[y:y+h, x:x+w, :]
        
        # Mengubah ukuran wajah yang dipotong menjadi 50x50 piksel
        resized_img = cv2.resize(crop_img, (50, 50))
        
        # Menyimpan gambar wajah ke dalam list (Maksimal 100 gambar, diambil setiap 10 frame)
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
            print(f"Menyimpan data wajah ke-{len(faces_data)}/100")
        
        i += 1
        
        # Menampilkan teks jumlah wajah yang sudah disimpan di layar
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        
        # Menggambar kotak merah di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        
    # Menampilkan hasil video ke layar
    cv2.imshow("Frame Pengenalan Wajah", frame)
    
    # Menunggu tombol ditekan selama 1 milidetik
    k = cv2.waitKey(1)
    
    # Jika tombol 'q' ditekan, program akan berhenti
    if k == ord('q'):
        print("Program dihentikan oleh pengguna.")
        break
    
    # Program otomatis berhenti jika sudah mengumpulkan 100 data wajah (opsional, sesuai modul)
    if len(faces_data) >= 100:
        print("Pengumpulan 100 data wajah selesai!")
        break

# Melepaskan Kamera dan menutup semua jendela UI OpenCV
video.release()
cv2.destroyAllWindows()
import pickle

# --- TAMBAHAN UNTUK MENYIMPAN DATA WAJAH ---
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Menggunakan nama 'richie' secara otomatis sesuai nama fotomu
nama = "richie" 

# Menyimpan Label/Nama
try:
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
except:
    names = []
names = names + [nama] * 100
with open('names.pkl', 'wb') as f:
    pickle.dump(names, f)

# Menyimpan Data Gambar Wajah
try:
    with open('faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
except:
    faces = faces_data

with open('faces_data.pkl', 'wb') as f:
    pickle.dump(faces, f)

print(f"Dataset wajah untuk {nama} berhasil dibuat dan disimpan!")