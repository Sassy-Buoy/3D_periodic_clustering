import numpy as np
import umap.umap_ as umap
#from sklearn.decomposition import PCA

encoded_data = np.load("dash_app_viz/latent_space_ae.npy")

# encoded_data: shape (1000, 2601, 5)
epochs, n_points, n_dims = encoded_data.shape
flat = encoded_data.reshape(-1, n_dims)               # (2_601_000, 5)

# Optional: PCA speedup (useful if n_dims >> 5; here optional)
# pca = PCA(n_components=min(20, n_dims), random_state=0)
# flat = pca.fit_transform(flat)

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
flat_2d = reducer.fit_transform(flat)                  # (2_601_000, 2)

embedded = flat_2d.reshape(epochs, n_points, 2)        # (1000, 2601, 2)

np.save("dash_app_viz/latent_space_ae_reduced.npy", embedded)
