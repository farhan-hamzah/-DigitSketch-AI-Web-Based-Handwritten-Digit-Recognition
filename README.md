---

# ✍️ DigitSketch AI: Web-Based Handwritten Digit Recognition

<div align="center">
  <em>DigitSketch AI adalah aplikasi web interaktif berbasis AI yang mampu mengenali digit tulisan tangan (0–9) secara real-time melalui kanvas gambar. Sistem ini menggabungkan kekuatan <strong>frontend modern</strong>, <strong>backend API dengan Go (Gin)</strong>, dan <strong>model deep learning CNN (TensorFlow)</strong>.</em>
</div>

---

## 🌟 Fitur Utama

- 🎨 Gambar angka langsung di canvas dengan mudah.
- 🤖 Prediksi angka secara real-time menggunakan model CNN.
- 📊 Tampilan confidence score untuk memastikan akurasi.
- ⚙️ Arsitektur modular dengan pemisahan Frontend, Backend, dan Model.
- 🔒 Akses aman hanya melalui lingkungan lokal (`localhost`).

---

## 🛠️ Teknologi yang Digunakan

| **Layer**         | **Teknologi**                  |
|-------------------|---------------------------------|
| **Frontend**      | HTML5, CSS3, JavaScript (Canvas API) |
| **Backend API**   | Go (Gin Framework)             |
| **Model Prediksi**| Python 3, TensorFlow, Keras    |
| **Komunikasi**    | JSON via REST API              |

---

## 📂 Struktur Proyek

```
learnTensorflow/
│
├── be.go              # Backend Go (Gin)
├── fe.html            # Frontend HTML
├── predict.py         # Python handler untuk prediksi
├── mnist_cnn_improved.keras # Model hasil training
├── testAngka/         # Folder penyimpanan input gambar dari canvas
│   └── digit.png      # File sementara dari gambar canvas
```

---

## 🔍 Cara Kerja

1. **Pengguna Menggambar**  
   Angka digambar langsung pada canvas menggunakan mouse atau touch.

2. **Konversi & Pengiriman**  
   Canvas dikonversi menjadi base64 PNG dan dikirim ke backend via endpoint `/predict`.

3. **Proses Backend**  
   Backend Go menyimpan gambar dan menjalankan `predict.py` menggunakan `exec.Command`.

4. **Prediksi Model**  
   Python melakukan preprocessing gambar, memuat model `.keras`, dan menghitung prediksi.

5. **Tampilan Hasil**  
   Prediksi angka dan confidence score dikembalikan ke frontend untuk ditampilkan.

---

## 🚀 Cara Menjalankan Proyek

### 1. Jalankan Backend (Go)
```bash
go run be.go
```
- Pastikan server berjalan di `http://localhost:8080`.

### 2. Jalankan Frontend
- Buka file `fe.html` melalui Live Server atau secara manual di:
  ```
  127.0.0.1:5500/fe.html
  ```

### 3. Persiapan Model & Python
- Pastikan model sudah dilatih dan disimpan sebagai `mnist_cnn_improved.keras`.
- Install dependensi Python:
  ```bash
  pip install tensorflow opencv-python matplotlib pillow numpy
  ```

### 🎯 Contoh Output dari API
```json
{
  "prediction": "4",
  "confidence": "0.91"
}
```

---

## ✨ Demo (Offline)
- Versi online tidak tersedia. Aplikasi hanya dapat diakses melalui `localhost`.

---

### 💡 Catatan
- Pastikan semua file berada di direktori yang sama.
- Gunakan konsol untuk debugging jika ada error (lihat log di browser atau terminal).

---

### 🎉 Kontribusi
Jika Anda ingin berkontribusi, silakan buka issue atau pull request di repository ini!

---

### Versi yang Diperbarui
- **Tanggal**: 08 Juli 2025, 21:42 WIB

---
