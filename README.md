## Richie Pranata
## 312410451
# Laporan Progres Tugas: Implementasi Sistem Deteksi dan Pengenalan Wajah Berbasis OpenCV

Laporan ini disusun untuk menjelaskan alur pengerjaan, kendala yang dihadapi, serta perbaikan teknis yang telah dilakukan pada file `Tugas1.py` dan `Tugas2.py` dalam rangka memenuhi tugas mata kuliah Pengolahan Citra Digital.

---

## 1. Alur Pengerjaan dan Perbaikan Kode

### A. Tugas 1: Deteksi Wajah dan Mata pada Citra Statis (`Tugas1.py`)
Pada tahap awal, program dirancang untuk mendeteksi komponen wajah dan mata menggunakan algoritma **Haar Cascade Classifier** dari OpenCV pada sebuah file gambar statis.

*   **Kendala yang Ditemukan:** 
    Saat pertama kali dijalankan, program mengalami kegagalan sistem berupa `AttributeError: 'NoneType' object has no attribute 'copy'`. Hal ini terjadi karena fungsi `cv2.imread()` tidak dapat menemukan atau membaca file gambar target akibat ketidaksesuaian nama file atau kesalahan jalur direktori (*working directory*).
*   **Perbaikan yang Dilakukan:**
    1. Melakukan pemeriksaan direktori kerja untuk memastikan file gambar berada dalam satu folder root yang sama dengan skrip Python.
    2. Mengubah penamaan file gambar pada kode dari `'ica.jpeg'` menjadi `'richie.jpeg'` untuk menyesuaikan dengan aset gambar yang tersedia di lokal komputer.
    3. Mengubah pendekatan pembacaan *path* file menggunakan jalur absolut (`C:/xampp/htdocs/Citra Richie/richie.jpeg`) untuk memastikan OpenCV dapat menemukan *resource* gambar secara akurat tanpa *ambiguity*.
*   **Hasil Akhir:** 
    Program berhasil mengeksekusi fungsi `adjusted_detect_face` dan `detect_eyes`. Output berupa tiga file gambar baru (`face.jpg`, `eyes.jpg`, dan `face+eyes.jpg`) berhasil digenerate dan disimpan ke dalam direktori proyek.

### B. Tugas 2: Pengumpulan Dataset Wajah via Webcam (`Tugas2.py`)
Program kedua ini berfungsi sebagai subsistem *Data Acquisition* (akuisisi data) yang bertugas mengumpulkan sampel wajah secara *real-time* melalui perangkat *webcam*.

*   **Evaluasi Kode Awal:**
    Kode awal berhasil mengaktifkan kamera, melakukan deteksi wajah secara *live*, dan memotong (*crop*) area wajah menjadi resolusi $50 \times 50$ piksel sebanyak 100 frame. Namun, data tersebut hanya ditampung di dalam memori RAM (`faces_data = []`) dan akan terhapus otomatis saat aplikasi ditutup karena belum ada fungsi *write* ke penyimpanan lokal.
*   **Perbaikan dan Pengembangan Kode:**
    Untuk menyambung alur kerja ke tahap pengenalan (*recognition*), dilakukan modifikasi pada akhir skrip dengan menambahkan modul `pickle`. 
    1. Mengubah struktur list data wajah menjadi format matriks (*NumPy array*) dan melakukan perataan dimensi (*reshape/flatten*).
    2. Menambahkan fungsi otomatisasi pelabelan dengan string nama `'richie'`.
    3. Mengimplementasikan mekanisme penyimpanan berbasis serialisasi data ke dalam file `names.pkl` (untuk menyimpan label data) dan `faces_data.pkl` (untuk menyimpan vektor fitur gambar wajah).
*   **Hasil Akhir:**
    Saat kamera dijalankan, sistem berhasil menangkap 100 sampel wajah secara sekuensial. Setelah kuota data terpenuhi, program menutup *resource* kamera dengan aman dan berhasil mengekspor dataset digital berupa file `.pkl` ke direktori lokal sebagai aset untuk fase *training* model berikutnya.

---

## 2. Alur Logika Sistem Secara Keseluruhan

Secara metodologi, proyek ini dibagi menjadi tiga tahapan utama:


```

[Tahap 1: Pengujian Algoritma] ──> [Tahap 2: Pengumpulan Dataset] ──> [Tahap 3: Model Pengenalan]
(Tugas1.py via Gambar)              (Tugas2.py via Webcam)             (Rencana Validasi KNN)

```

1.  **Fase Validasi Deteksi (Tugas 1):** Memastikan bahwa parameter *Haar Cascade* (`scaleFactor` dan `minNeighbors`) sudah optimal dan mampu mengenali fitur wajah serta mata objek dengan akurat pada media dua dimensi.
2.  **Fase Ekstraksi Fitur & Dataset (Tugas 2):** Mengambil sampel variasi wajah objek (posisi, pencahayaan, atau ekspresi mikro) lewat kamera interaktif untuk dijadikan basis data pengenal.
3.  **Fase Komputasi Pengenalan (Next Step):** Dataset `.pkl` yang telah berhasil disimpan pada Tugas 2 nantinya siap diumpankan ke dalam algoritma klasifikasi (seperti *K-Nearest Neighbors* atau *Support Vector Machine*) untuk melakukan pencocokan wajah (*Face Recognition*) secara langsung dan menampilkan label nama objek di atas layar monitor.

```

Hasil Laporan :
<img width="941" height="466" alt="Screenshot 2026-06-05 021939" src="https://github.com/user-attachments/assets/3475d6de-99c4-493a-9f05-abf56bfce1db" />
