# TrueShade AI 💄✨

**TrueShade AI** adalah aplikasi web cerdas yang menggunakan *Machine Learning* untuk menganalisis warna kulit (skin tone) pengguna secara real-time dan memberikan rekomendasi produk kosmetik yang paling cocok.

Dibangun menggunakan **React** untuk antarmuka yang responsif dan **TensorFlow.js** untuk pemrosesan AI langsung di browser (Client-side), menjaga privasi pengguna karena foto tidak perlu dikirim ke server.

![TrueShade Banner](https://via.placeholder.com/1200x600?text=TrueShade+AI+Preview)
*(Ganti link di atas dengan screenshot aplikasi Anda nantinya)*

## 🌟 Fitur Utama

* **📸 Analisis Wajah Real-time:** Mendeteksi tone warna kulit menggunakan kamera perangkat atau upload foto galeri.
* **🤖 AI Powered:** Menggunakan model *Deep Learning* (MobileNetV2) yang telah di-finetune untuk klasifikasi warna kulit.
* **🛍️ Rekomendasi Cerdas:** Menyarankan produk kosmetik (Foundation, Lipstik, dll) berdasarkan harmoni warna (Color Theory).
* **📱 Desain Responsif:** Tampilan yang optimal baik di Desktop maupun Mobile.
* **⚡ Cepat & Privat:** Semua proses analisis dilakukan di browser pengguna menggunakan TensorFlow.js.

## 🛠️ Teknologi yang Digunakan

* **Frontend:** [React](https://reactjs.org/) + [Vite](https://vitejs.dev/)
* **Styling:** [Tailwind CSS](https://tailwindcss.com/)
* **Machine Learning:** [TensorFlow.js](https://www.tensorflow.org/js)
* **Icons:** [Lucide React](https://lucide.dev/)
* **Dataset:** Makeup API / Female Daily (Custom JSON Dataset)

## 🚀 Cara Menjalankan Project (Instalasi)

Ikuti langkah-langkah ini untuk menjalankan proyek di komputer lokal Anda:

### 1. Clone Repository
```bash
git clone [https://github.com/username-anda/trueshade-ai.git](https://github.com/username-anda/trueshade-ai.git)
cd trueshade-ai
