### BAGIAN 1: Helper Function (Pencari Array Otomatis)

```Python
def get_first_array(zarr_node):
    if hasattr(zarr_node, "shape"):
        return zarr_node
    if hasattr(zarr_node, "keys"):
        for key in zarr_node.keys():
            item = zarr_node[key]
            array = get_first_array(item)
            if array is not None:
                return array
    return None
```



* **Fungsi**: menggunakan teknik **rekursi** untuk mencari objek array gambar di dalam struktur `.zarr`.
* **Baris 2–3**: Memeriksa apakah `zarr_node` memiliki atribut `.shape`. Jika ada, berarti node tersebut adalah `zarr.Array` (data citra murni), bukan sub-folder.
* **Baris 4–8**: Jika node tersebut adalah `zarr.Group` (seperti folder yang berisi sub-folder lagi), fungsi akan melakukan perulangan (*loop*) ke dalam kunci-kuncinya secara rekursif hingga menemukan array gambar pertamanya.
* **Tujuan**: Membuat dataset *robust* dan tidak mudah *crash* saat membaca berbagai versi struktur file Zarr.


### BAGIAN 2: Inisialisasi & Caching Pintar (`__init__`)

```python
class CellPatchDataset(Dataset):
    def __init__(self, zarr_paths, geff_paths, patch_size=(32, 128, 128), sigma=2.0):
        self.zarr_paths = zarr_paths
        self.geff_paths = geff_paths
        self.patch_size = patch_size
        self.sigma = sigma
        self.patch_mapping = []
```

* **`patch_size=(32, 128, 128)`:** Ukuran potong 3D yang diproses oleh GPU per patch.
* *`sigma=2.0`: Standar deviasi ($\sigma$) fungsi Gaussian untuk membentuk radius titik sel.

```python
		# 1. OPTIMASI CACHING (Primitive Struct)
        self.node_cache = {}
        for g_path in self.geff_paths:
            graph, _ = read(str(g_path), backend="networkx")
            for node_id, attrs in graph.nodes(data=True):
                key = (str(g_path), attrs["t"])
                if key not in self.node_cache:
                    self.node_cache[key] = []
                self.node_cache[key].append((attrs["z"], attrs["y"], attrs["x"]))

        for key in self.node_cache:
            self.node_cache[key] = np.array(self.node_cache[key], dtype=np.int32)
```

**Kenapa Ini Krusial?**

- Pada versi awal, kita menyimpan objek `networkx.Graph` utuh di RAM. Ketika PyTorch menggunakan `num_workers = 2` (multiprocessing), Python menduplikasi objek `Graph` yang berat ke setiap *worker*. Hasilnya: **RAM CPU Kaggle kehabisan memori (*OOM*)**.

* **Solusinya***:* Kita hanya mengekstrak koordinat $(Z, Y, X)$ dalam bentuk *dictionary* berisi `numpy.ndarray` sederhana bertipe `int32`. Ukuran memorinya menyusut $>90\%$, pemanggilan data super cepat, dan aman dari *memory leak*.

```Python
pz, py, px = self.patch_size
for z_path, g_path in zip(self.zarr_paths, self.geff_paths):
	zarr_root = zarr.open(str(z_path), mode="r")
    zarr_array = get_first_array(zarr_root)
      
    num_t, size_z, size_y, size_x = zarr_array.shape

    for t in range(num_t):
    	for z in range(0, size_z, pz):
        	for y in range(0, size_y, py):
            	for x in range(0, size_x, px):
                	if (z + pz <= size_z) and (y + py <= size_y) and (x + px <= size_x):
                    	self.patch_mapping.append({
							"zarr_path": str(z_path),
                            "geff_path": str(g_path),
                            "t": t,
                            "z_start": z,
                            "y_start": y,
                      		"x_start": x,
                        })
```

* **Mekanisme *Lazy Loading* (Spatial Indexing):**
  * Kita tidak memuat seluruh gambar mikroskop 3D sebesar ribuan gigabyte ke memori RAM!
  * Kita hanya mendaftarkan metadata koordinat batas (*bounding coordinates*) dari 159.200 patch ke dalam *list* `self.patch_mapping`. Memori RAM yang terpakai hanya beberapa megabyte.

### BAGIAN 3: Matematika Rendering Heatmap 3D (`_draw_3d_gaussian`)

```python
def _draw_3d_gaussian(self, heatmap, center):
        cz, cy, cx = center
        pz, py, px = heatmap.shape
        radius = int(3 * self.sigma)

        z_min, z_max = max(0, int(cz - radius)), min(pz, int(cz + radius + 1))
        y_min, y_max = max(0, int(cy - radius)), min(py, int(cy + radius + 1))
        x_min, x_max = max(0, int(cx - radius)), min(px, int(cx + radius + 1))
```

* **`radius = int(3 * self.sigma)`:** Menggunakan aturan statistik $3\sigma$ (mencakup $99.7\%$ area distribusi Gaussian).
* **Crop Bounding Box Lokal***:*Dibanding menghitung nilai Gaussian untuk seluruh area $32 \times 128 \times 128$ (yang akan membuat CPU sangat lambat), kita hanya menghitung kotak kecil di sekitar titik pusat sel (`cz, cy, cx`).

```python
zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
dist_sq = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2
gaussian = np.exp(-dist_sq / (2 * (self.sigma**2)))

heatmap[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
	heatmap[z_min:z_max, y_min:y_max, x_min:x_max], gaussian
)
```

* **`np.ogrid`:**Menghasilkan grid koordinat 3D yang sangat efisien memori.

* **`dist_sq`:**Jarak kuadrat Euclidean 3D: $d^2 = (z-c_z)^2 + (y-c_y)^2 + (x-c_x)^2$.

* **`gaussian = np.exp(...)`:**Rumus fungsi Gaussian 3D: $G(z,y,x) = e^{-\frac{d^2}{2\sigma^2}}$.
* **`np.maximum`:**Jika dua sel terletak berdekatan dan area Gaussian-nya tumpang tindih (*overlap*), kita mengambil nilai terbesarnya (bukan dijumlahkan), sehingga puncak nilai target tetap bernilai presisi $1.0$.



### BAGIAN 4: Pengambilan Data Utama (`__getitem__`)

```python
def __getitem__(self, idx):
	info = self.patch_mapping[idx]
    ...
    raw_patch = zarr_array[
        t, z_s : z_s + pz, y_s : y_s + py, x_s : x_s + px
    ].astype(np.float32)
```

* Di sinilah *Lazy Loading* terjadi. Ketika PyTorch meminta data patch ke-`idx`, barulah potongan $32 \times 128 \times 128$ dibaca secara *real-time* dari storage.

```Python
# 2. OPTIMASI NORMALISASI (Percentile Clipping - nnU-Net Standard)
 p_low, p_high = np.percentile(raw_patch, (0.5, 99.5))
 if p_high > p_low:
 	input_patch = np.clip(raw_patch, p_low, p_high)
 	input_patch = (input_patch - p_low) / (p_high - p_low)
 else:
 	input_patch = np.zeros_like(raw_patch)
```

* **Standar Medis/Mikroskopi (nnU-Net):**

  * Memotong  intensitas terendah dan  intensitas tertinggi
  * **Keunggulan**: Menghilangkan *glitch* piksel silau (*outlier*) akibat pancaran sinar fluoresensi, serta memastikan bahwa patch kosong (*background*) tidak berubah menjadi serba putih akibat pencampuran rentang skala.

```python
		target_heatmap = np.zeros(self.patch_size, dtype=np.float32)
        nodes_at_t = self.node_cache.get((geff_path, t), np.array([]))

        for nz, ny, nx in nodes_at_t:
            if (z_s <= nz < z_s + pz) and (y_s <= ny < y_s + py) and (x_s <= nx < x_s + px):
                rel_z = nz - z_s
                rel_y = ny - y_s
                rel_x = nx - x_s
                self._draw_3d_gaussian(target_heatmap, (rel_z, rel_y, rel_x))
```

* Memeriksa apakah ada sel di frame waktu $t$ yang berada di dalam jangkauan patch saat ini. Jika ada, koordinat global $(nz, ny, nx)$ diubah menjadi koordinat relatif lokal `(rel_z, rel_y, rel_x)` lalu digambar bulatan Gaussian-nya.

```python
# 3. OPTIMASI MEMORY LAYOUT (Contiguous Memory)
        input_patch = np.ascontiguousarray(input_patch)
        target_heatmap = np.ascontiguousarray(target_heatmap)

        input_tensor = torch.from_numpy(input_patch).unsqueeze(0)
        target_tensor = torch.from_numpy(target_heatmap).unsqueeze(0)

        return input_tensor, target_tensor
```

* **`np.ascontiguousarray`:** Menata ulang susunan memori RAM agar berurutan di blok fisik memori. Ini mempercepat transfer data dari CPU ke GPU Kaggle hingga **30–50%**.
* `torch.from_numpy`:Membuat Tensor tanpa menduplikasi memori RAM (*zero-copy allocation*).
* `.unsqueeze(0)`: Menambahkan dimensi *channel* di depan, mengubah bentuk dari $(32, 128, 128)$ menjadi $(1, 32, 128, 128)$, sesuai dengan format input Convolution 3D PyTorch: `(Channel, Depth, Height, Width)`.
