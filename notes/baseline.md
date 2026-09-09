Processing a time-series microscopy volume into a spatial-temporal graph involves two distinct phases: **Instance Detection** (turning raw intensity voxels into single-cell coordinate nodes) and **Data Association** (tracking those nodes over time to build edge trajectories).

Here is a breakdown of the strategy, math, and code implementation to complete your function.

---

### **Strategy & Logic**

#### **1. Node Detection Strategy**

* **Threshold & Bounding Box Filtering:** Raw intensity data is converted into binary masks using thresholding (Otsu or absolute intensity). `scipy.ndimage.label` groups adjacent non-zero voxels into distinct contiguous 3D regions (connected components).
* **Centroid Calculation:** `scipy.ndimage.center_of_mass` computes the center of intensity or spatial volume $(\bar{z}, \bar{y}, \bar{x})$ for each connected component. Regions that are too small (noise) or too large (clumped artifacts) are filtered out using volume thresholds.

#### **2. Temporal Linkage Strategy**

* **Anisotropic Scaling:** Microscopes rarely have isotropic resolution. Z-planes are typically spaced further apart than X/Y pixels (e.g., $dz = 1.0\,\mu\text{m}$, $dx = dy = 0.2\,\mu\text{m}$). Before computing physical distance, convert voxel coordinates to physical units ($\mu\text{m}$):

$$Z_{\text{real}} = z \times dz, \quad Y_{\text{real}} = y \times dy, \quad X_{\text{real}} = x \times dx$$


* **Spatial Tracking (Gating + Matching):**
1. Use `scipy.spatial.KDTree` or `NearestNeighbors` to quickly query spatial proximity within a physical distance cutoff ($< 0.7\,\mu\text{m}$).
2. For a global 1-to-1 optimum, construct a distance matrix between nodes at frame $t$ and frame $t+1$, mask out pairs exceeding $0.7\,\mu\text{m}$ with infinite cost, and solve the linear assignment problem (`scipy.optimize.linear_sum_assignment`).



---

### **Implementation**

```python
import numpy as np
import zarr
from scipy.ndimage import label, center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu

def process_dataset(dataset_name, zarr_root, voxel_spacing=(1.0, 0.25, 0.25), max_distance_um=0.7):
    """
    Processes a 4D Zarr array into spatial nodes and temporal edges.
    
    Parameters:
    -----------
    dataset_name : str
        Identifier for the dataset.
    zarr_root : zarr.Array or dask.array
        4D array indexed by (t, z, y, x).
    voxel_spacing : tuple of float
        Physical size per voxel along (Z, Y, X) axes in micrometers.
    max_distance_um : float
        Maximum allowed distance threshold for linking nodes between adjacent frames.
    """
    nodes = []
    edges = []

    # Metadata dimensions
    total_frames = zarr_root.shape[0]
    dz, dy, dx = voxel_spacing

    # Dictionary mapping time frame -> list of node IDs created at frame t
    nodes_by_frame = {}
    node_id_counter = 0

    # -------------------------------------------------------------------------
    # PHASE 1: NODE DETECTION
    # -------------------------------------------------------------------------
    for t in range(total_frames):
        # Extract 3D volume lazily from Zarr
        volume = np.array(zarr_root[t])
        
        # 1. Segment volume (Thresholding)
        # Using simple Otsu or fallback intensity threshold
        thresh = threshold_otsu(volume) if volume.max() > 0 else 0
        binary_mask = volume > thresh
        
        # 2. Label 3D connected components
        labeled_volume, num_features = label(binary_mask)
        
        # 3. Extract centroids if features are detected
        frame_node_ids = []
        if num_features > 0:
            # Get centroids in voxel coordinates (z, y, x)
            centroids_voxel = center_of_mass(binary_mask, labeled_volume, range(1, num_features + 1))
            
            for index, (z, y, x) in enumerate(centroids_voxel):
                # Convert to physical coordinates (micrometers)
                z_um, y_um, x_um = z * dz, y * dy, x * dx
                
                node_info = {
                    "id": node_id_counter,
                    "frame": t,
                    "voxel_coords": (float(z), float(y), float(x)),
                    "physical_coords": (float(z_um), float(y_um), float(x_um))
                }
                
                nodes.append(node_info)
                frame_node_ids.append(node_id_counter)
                node_id_counter += 1
                
        nodes_by_frame[t] = frame_node_ids

    # Fast lookup table for physical coordinates by node ID
    node_lookup = {node["id"]: node["physical_coords"] for node in nodes}

    # -------------------------------------------------------------------------
    # PHASE 2: TEMPORAL LINKAGE
    # -------------------------------------------------------------------------
    for t in range(total_frames - 1):
        ids_t = nodes_by_frame[t]
        ids_t_next = nodes_by_frame[t + 1]

        if not ids_t or not ids_t_next:
            continue

        # Extract coordinate matrices shape (N, 3) and (M, 3)
        coords_t = np.array([node_lookup[nid] for nid in ids_t])
        coords_t_next = np.array([node_lookup[nid] for nid in ids_t_next])

        # Calculate Euclidean physical distance matrix (N x M)
        dist_matrix = cdist(coords_t, coords_t_next, metric='euclidean')

        # Apply maximum distance gating cutoff
        cost_matrix = dist_matrix.copy()
        cost_matrix[dist_matrix > max_distance_um] = 1e9  # Set invalid links to high cost

        # Solve optimal 1-to-1 bipartite matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter out links that exceeded the max_distance_um threshold
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= max_distance_um:
                source_node_id = ids_t[r]
                target_node_id = ids_t_next[c]
                edges.append((source_node_id, target_node_id))

    return nodes, edges

```

---

### **How to Run Across Your 4 Local Zarr Datasets**

Since you mentioned you have four Zarr folders inside a local test folder, you can iterate over them using `pathlib` and open each level `0` resolution array:

```python
from pathlib import Path

test_folder = Path("./path/to/test_folder")

results = {}
for zarr_path in test_folder.glob("*.zarr"):
    dataset_name = zarr_path.stem
    
    # Open level '0' array from Zarr hierarchy
    zarr_root = zarr.open(str(zarr_path / "0"), mode='r')
    
    print(f"Processing {dataset_name} with shape {zarr_root.shape}...")
    nodes, edges = process_dataset(dataset_name, zarr_root)
    
    results[dataset_name] = {"nodes": nodes, "edges": edges}
    print(f"Done: {len(nodes)} nodes, {len(edges)} edges detected.")

```

---

### **Key Considerations**

1. **Voxel Spacing Tuning:** Ensure `voxel_spacing=(dz, dy, dx)` matches your actual microscope calibration metadata. If distance is calculated strictly in voxel coordinates without physical scaling, the $0.7\,\mu\text{m}$ threshold will filter out valid movements along Z due to anisotropic pixel resolution.
2. **Memory Efficiency:** Indexing `zarr_root[t]` extracts only 1 volume at a time into RAM, keeping the process lightweight even for multi-gigabyte files.
3. **Advanced Detection:** If simple thresholding over-segments or misses dim cells, swap `threshold_otsu` and `label` with 3D Laplacian of Gaussian (LoG) blob detection via `skimage.feature.blob_log` or a trained deep learning segmenter like `cellpose`.