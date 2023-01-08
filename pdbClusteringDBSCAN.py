import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from Bio.PDB import PDBParser, Superimposer

# Specify the folder containing the PDB files
folder = "pdb_files"

# Load the first PDB file
parser = PDBParser()
structure = parser.get_structure("structure", os.path.join(folder, "1.pdb"))
# Extract the coordinates of the alpha carbon atoms
ref_alpha_carbons = []
for residue in structure.get_residues():
    ref_alpha_carbons.append(residue["CA"].get_coord())
ref_alpha_carbons = np.array(ref_alpha_carbons)

# Initialize a list to store the feature vectors for each PDB file
pdb_files = []
for pdb_file in os.listdir(folder):
    if pdb_file.endswith(".pdb"):
        # Load the PDB file
        structure = parser.get_structure("structure", os.path.join(folder, pdb_file))
        # Extract the coordinates of the alpha carbon atoms
        alpha_carbons = []
        for residue in structure.get_residues():
            alpha_carbons.append(residue["CA"].get_coord())
        alpha_carbons = np.array(alpha_carbons)
        # Superimpose the structure on the reference structure
        superimposer = Superimposer()
        superimposer.set_atoms(ref_alpha_carbons, alpha_carbons)
        alpha_carbons = superimposer.apply(alpha_carbons)
        # Flatten the coordinates into a single feature vector
        pdb_files.append(alpha_carbons.flatten())

# Standardize the feature vectors
scaler = StandardScaler()
pdb_files_scaled = scaler.fit_transform(pdb_files)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(pdb_files_scaled)

# Print the cluster labels
print(clusters)
