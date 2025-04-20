import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def main(input_path, cluster_path):
    # Load PCA data and clustering results
    input_df = pd.read_csv(input_path)
    input_df['id'] = input_df.index
    cluster_df = pd.read_csv(cluster_path)

    # Merge on 'id'
    df = input_df.merge(cluster_df, on='id')

    # Coordinates and labels
    x = df['x']
    y = df['y']
    z = df['z']
    c = df['cluster']

    # Create 3D scatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=c, s=1)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('cluster')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('K‑means Clusters in 3D PCA Space')

    # Define view angles: (elevation, azimuth)
    angles = [
        (30, 45),   # Default perspective
        (90, 0),    # Top-down view
        (0, 0),     # Side view
        (0, 90),    # Other side
    ]

    # Save one image per angle
    for elev, azim in angles:
        ax.view_init(elev=elev, azim=azim)
        fname = f"clusters_e{elev}_a{azim}.png"
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved {fname}")

    plt.close(fig)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python visualize.py <input_csv> <cluster_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
