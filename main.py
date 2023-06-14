from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse, os
import random
from sklearn.decomposition import PCA
import esm

def seed_all(seed=42):
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None

def load_structure(pdb_file):
    chain_id = 'A'
    structure = esm.inverse_folding.util.load_structure(pdb_file, chain_id)
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    # print('Native sequence:')
    # print(native_seq)

    return coords

def load_fasta(fasta_file='data/covid-mutated-seqs/RBD (331-528 aa) alignment.fasta'):
    seq_dict = {}
    records = SeqIO.parse(fasta_file, 'fasta')
    for rec in records:
        seq_dict[str(rec.id)] = str(rec.seq)
    # print([len(seq) for seq in seq_dict.values()])
    print(f'Loaded {len(seq_dict)} sequences.')
        
    return seq_dict

def run_omegafold(fasta_file, output_dir='output_omegafold'):
    print(f'### Generating OmegaFold structure predictions...')
    os.system(f'omegafold {fasta_file} {output_dir}')

def run_esm1b(fasta_file, output_dir='esm1b_emb'):
    print(f'### Generating ESM-1b embeddings')
    os.system(f'python extract.py esm1b_t33_650M_UR50S {fasta_file} {output_dir} --include per_tok')

def run_esmif(pdb_dir='output_omegafold', output_dir='esmif_emb'):
    print(f'### Generating ESM-IF embeddings')
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(pdb_dir):
        files = [os.path.join(root, file) for file in files]
    for pdb_file in files:
        coords = load_structure(pdb_file)
        # print(coords)
        rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords).detach().cpu()
        print(len(coords), rep.shape)
        torch.save(rep, os.path.join(output_dir, f'{os.path.basename(pdb_file)[:-4]}.pt'))
    

def get_esm1b_esmif_emb(sid):
    esm1b_dir = 'esm1b_emb'
    esm1b_seq = torch.load(os.path.join(esm1b_dir, sid+'.pt'))['representations'][33]
    esmif_dir = 'esmif_emb'
    esmif_seq = torch.load(os.path.join(esmif_dir, sid+'.pt'))
    
    emb_seq = torch.cat([esm1b_seq, esmif_seq], dim=1)
    
    return emb_seq

def plot_pca(embeddings, ids, output_file):
    print(f'### Performing PCA and generating visulizations...')
    embeddings = [np.array(emb) for emb in embeddings]
    X = np.vstack(embeddings)
    # Perform PCA on the embeddings
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    # Create a scatter plot of the reduced embeddings
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the embeddings as a scatter plot
    colors = range(len(embeddings))
    ax.scatter(x, y, s=400, c=colors, cmap='jet')
    # Add annotations for each point
    for i, id in enumerate(ids):
        ax.annotate(id, (x[i], y[i]))
    ax.axis('off')
    ax.set_title(os.path.basename(output_file)[:-4])
    # Save the figure to output file
    plt.savefig(output_file, dpi=400)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fasta_file', '-i', type=str, default='covid-RBD.fasta')
    # parser.add_argument('--dist', '-d', type=str, choices=['hamming', 'blosum62', 'one_hot', 'node2vec', 'esm1b', 'mp', 'esmif', 'esm1b_esmif'])
    # parser.add_argument('--plot', '-p', type=str, default='networkx', choices=['networkx', 'mds', 'tsne', 'pca', 'umap'])
    parser.add_argument('--output', '-o', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default='visual_results')
    parser.add_argument('--seed', '-s', type=int, default=42)
    # parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.cache:
        # run_omegafold(args.fasta_file)
        # run_esm1b(args.fasta_file)
        run_esmif()
    
    seq_dict = load_fasta(args.fasta_file)
    seq_list = list(seq_dict.values())
    seq_ids = list(seq_dict.keys())
    
    embeddings = []
    for sid in seq_ids:
        embeddings.append(get_esm1b_esmif_emb(sid).mean(dim=1))
    plot_pca(embeddings, seq_ids, os.path.join(args.out_dir, f"{args.output}-esm1b_esmif-pca.png"))
    