
1. Mesin Dasar: `_double_conv` (Blok Lego Utama)

Sebelum merakit keseluruhan pabrik, kita membuat blok fondasinya. Hampir seluruh bagian U-Net dibangun menggunakan blok ini secara berulang.

* **`nn.Conv3d` (Konvolusi):** Memindai gambar 3D menggunakan kotak kecil (kernel 3x3x3) untuk mencari pola visual seperti tepi, sudut, atau tekstur sel. `padding=1` memastikan ukuran gambar (Z, Y, X) tidak menyusut setelah dipindai.
* **`nn.BatchNorm3d`:** Menstabilkan angka-angka hasil konvolusi agar tidak terlalu besar atau terlalu kecil. Ini membuat proses *training* jauh lebih cepat dan stabil.
* **`nn.ReLU`:** Fungsi aktivasi yang mengubah semua angka negatif menjadi nol. Ini memberi jaringan kemampuan untuk memahami pola non-linear (karena sel di dunia nyata bentuknya organik, bukan sekadar garis lurus).
* **Kenapa "Double"?** Melakukan konvolusi dua kali berturut-turut terbukti secara riset membuat model mampu mengekstrak fitur yang lebih kompleks dibandingkan hanya satu kali.

## 2. Persiapan Alat: `__init__`

Di dalam fungsi `__init__`, kita "membeli" dan menyiapkan semua mesin yang dibutuhkan. Jaringan saraf belum berjalan di sini.

* **Encoder (Jalur Turun):**
* Tugasnya adalah mengekstrak **konteks** ("Apakah ini bentuk sel?").
* Kita menyiapkan `self.enc1` dan `self.enc2`. Perhatikan jumlah filternya naik (1 -> 16 -> 32). Model semakin dalam, semakin banyak pola yang ia pelajari.
* `nn.MaxPool3d(kernel_size=2)`: Mesin ini memampatkan gambar menjadi setengah ukurannya (resolusi turun). Tujuannya agar model bisa melihat gambaran yang lebih luas tanpa kehabisan memori.
* **Bottleneck (Dasar U):**
* Titik terdalam model (`32 -> 64`). Di sini resolusi gambar paling kecil, tetapi informasi yang terkandung paling padat. Model sangat tahu *apa* yang ia lihat, tapi kehilangan informasi detail tentang *di mana* lokasi persisnya.
* **Decoder (Jalur Naik):**
* Tugasnya adalah mengembalikan **lokasi spasial** ("Di mana tepatnya koordinat sel ini?").
* `nn.ConvTranspose3d`: Ini adalah kebalikan dari MaxPool. Ia memperbesar kembali (upsample) resolusi gambar yang tadi sudah dimampatkan.
* Jumlah filter mulai diturunkan kembali (64 -> 32 -> 16).

## 3. Alur Pabrik: `forward`

Fungsi `forward` adalah tempat di mana tensor `x` (data gambar Zarr) mulai mengalir melewati mesin-mesin yang disiapkan di `__init__`. Bentuk arsitektur "U" terjadi di sini.

**Langkah 1: Menyimpan Memori Lokasi (Encoder)**
Saat data melewati `enc1` menjadi `x1`, kita **menyimpan `x1**`. Lalu data dikecilkan oleh `pool1` dan masuk ke `enc2` menjadi `x2`. Kita **menyimpan `x2**` lagi.
*Insight:* `x1` dan `x2` menyimpan informasi detail tentang letak piksel asli sebelum gambar diperkecil.

**Langkah 2: Memahami Makna (Bottleneck)**
Data melewati dasar U (`bn = self.bottleneck(p2)`). Di sini model memahami struktur sel secara makro, lalu mulai naik ke tahap Decoder.

**Langkah 3: Menyatukan Konteks dan Lokasi (Skip Connections)**
Inilah rahasia utama kehebatan U-Net:

1. **`up2`:** Gambar dari Bottleneck dibesarkan resolusinya.
2. **`torch.cat([up2, x2], dim=1)`:** Gambar yang baru dibesarkan (`up2`) **ditempelkan** dengan gambar memori dari masa lalu (`x2`).

* Dimensi tensor adalah `[Batch, Channel, Z, Y, X]`. `dim=1` berarti kita menumpuknya di dimensi *Channel*. Jika `up2` punya 32 channel dan `x2` punya 32 channel, hasil gabungannya memiliki 64 channel.
* Proses ini ibarat menggabungkan ingatan masa lalu ("Di sini letak sudut selnya") dengan pemahaman baru ("Oh, ini adalah selaput sel yang sedang membelah").

3. Proses ini diulang pada lapisan atasnya menggunakan `up1` dan `x1`.

**Langkah 4: Sentuhan Akhir (Final Conv)**
Di akhir Decoder, kita memiliki tensor dengan 16 channel. Padahal target *heatmap* kita hanya butuh 1 channel (hitam-putih).

* `nn.Conv3d(16, out_channels, kernel_size=1)`: Konvolusi berukuran 1x1x1 ini bertugas "memeras" 16 channel tersebut menjadi tepat 1 channel tanpa merusak ukuran spasial (Z, Y, X) sama sekali.
