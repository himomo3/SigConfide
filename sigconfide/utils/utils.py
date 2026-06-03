import numpy as np
import os

def FrobeniusNorm(M, P, E):
    return np.sqrt(np.sum((M - np.dot(P, E))**2))

def kl_divergence(m, m_approx):
    """Calculate the generalized Kullback-Leibler divergence for Poisson data."""
    m = np.asarray(m)
    m_approx = np.asarray(m_approx)
    eps = 1e-10
    return np.sum(m * np.log((m + eps) / (m_approx + eps)) - m + m_approx)


def is_wholenumber(x, tol=1e-15):
    return np.abs(x - np.round(x)) < tol

def detect_format(line):
    # Check if the line contains tabs - this suggests TSV format
    if '\t' in line:
        # Additionally, check if square brackets are present
        if '[' in line and (']' in line):
            return 'Mutated TSV Format', '\t'
        return 'TSV Format', '\t'

    # Check if the line contains commas - this suggests CSV format
    elif ',' in line:
        if ('[' in line) and (']' in line):
            return 'Mutated CSV Format', ','
        return 'CSV Format', ','

    # If none of the above were detected, return an unknown value
    return 'Unknown Format', None

def standardize_mutation_type(mutation, trinucleotide=None):
    mutation = mutation.strip('\'" ')
    if trinucleotide:
        trinucleotide = trinucleotide.strip('\'" ')
        if len(trinucleotide) == 3 and '>' in mutation:
            return f"{trinucleotide[0]}[{mutation}]{trinucleotide[2]}"
    return mutation

def get_mutation_types_from_file(file_name, format, sep):
    import csv
    with open(file_name, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=sep if sep else ',')
        try:
            header = next(reader)
        except StopIteration:
            return []
        header = [h.strip('\'" ') for h in header]
        
        has_trinucleotide = False
        mut_col_idx = 0
        trinuc_col_idx = -1
        
        for idx, col in enumerate(header):
            col_lower = col.lower()
            if col_lower in ['mutation type', 'mutation_type', 'type', 'mutation']:
                mut_col_idx = idx
            elif col_lower in ['trinucleotide', 'trinuc']:
                trinuc_col_idx = idx
                has_trinucleotide = True
                
        mutation_types = []
        for row in reader:
            if not row or not ''.join(row).strip():
                continue
            while len(row) <= max(mut_col_idx, trinuc_col_idx):
                row.append('')
            mut = row[mut_col_idx]
            trinuc = row[trinuc_col_idx] if has_trinucleotide else None
            std_type = standardize_mutation_type(mut, trinuc)
            mutation_types.append(std_type)
            
        return mutation_types

def sort_matrix_rows(matrix, mutation_types):
    num_types = len(mutation_types)
    std_types_list = None
    if num_types == 96:
        # SBS
        std_types_list = []
        for base5 in ['A', 'C', 'G', 'T']:
            for mut in ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']:
                for base3 in ['A', 'C', 'G', 'T']:
                    std_types_list.append(f"{base5}[{mut}]{base3}")
    elif num_types == 78:
        # DBS
        std_types_list = sorted(mutation_types)
    elif num_types == 83:
        # ID
        std_types_list = []
        for indel in ['Del', 'Ins']:
            for base in ['C', 'T']:
                for length in range(6):
                    std_types_list.append(f"1:{indel}:{base}:{length}")
        for indel in ['Del', 'Ins']:
            for size in [2, 3, 4, 5]:
                for length in range(6):
                    std_types_list.append(f"{size}:{indel}:R:{length}")
        for size in [2, 3, 4, 5]:
            limit = size if size == 5 else size - 1
            for length in range(1, limit + 1):
                std_types_list.append(f"{size}:Del:M:{length}")
    elif num_types == 48:
        # CN
        std_types_list = []
        for length in ['0-100kb', '100kb-1Mb', '>1Mb']:
            std_types_list.append(f"0:homdel:{length}")
        for state in ['1', '2', '3-4', '5-8', '9+']:
            for length in ['0-100kb', '100kb-1Mb', '1Mb-10Mb', '10Mb-40Mb', '>40Mb']:
                std_types_list.append(f"{state}:LOH:{length}")
        for state in ['2', '3-4', '5-8', '9+']:
            for length in ['0-100kb', '100kb-1Mb', '1Mb-10Mb', '10Mb-40Mb', '>40Mb']:
                std_types_list.append(f"{state}:het:{length}")
                
    if std_types_list:
        type_to_idx = {t: i for i, t in enumerate(mutation_types)}
        reorder_indices = []
        mismatch = False
        for t in std_types_list:
            if t in type_to_idx:
                reorder_indices.append(type_to_idx[t])
            else:
                t_lower = t.lower()
                found = False
                for p_t, idx in type_to_idx.items():
                    if p_t.lower() == t_lower:
                        reorder_indices.append(idx)
                        found = True
                        break
                if not found:
                    mismatch = True
                    break
                    
        if not mismatch and len(reorder_indices) == num_types:
            return matrix[reorder_indices, :]
            
    return matrix

def load_samples_file(file_name):
    with open(file_name, 'r') as file:
        csv_line = ''.join(file.readlines()).strip()
        format, sep = detect_format(csv_line)
        file.seek(0)
    if format == 'Unknown Format':
        raise ValueError('Unknown Format')

    samples = np.genfromtxt(file_name, delimiter=sep, skip_header=1)
    patient_names = np.genfromtxt(file_name, delimiter=sep, skip_header=0, max_rows=1, dtype=str).squeeze()

    if format == 'TSV Format' or format == 'Mutated TSV Format':
        samples = np.delete(samples, 0, axis=1)
        patients_names = patient_names[1:]

    if format == 'CSV Format' or format == 'Mutated CSV Format':
        samples = np.delete(samples, [0, 1], axis=1)
        patients_names = patient_names[2:]

    # Reorder samples to standard mutation type order if appropriate
    try:
        mutation_types = get_mutation_types_from_file(file_name, format, sep)
        if len(mutation_types) == samples.shape[0]:
            samples = sort_matrix_rows(samples, mutation_types)
    except Exception:
        pass

    return samples, patients_names

def load_signatures_file(file_name):
    with open(file_name, 'r') as file:
        csv_line = ''.join(file.readlines()).strip()
        format, sep = detect_format(csv_line)
        file.seek(0)
    signatures = np.genfromtxt(file_name, delimiter=sep, skip_header=1)
    names_signatures = np.genfromtxt(file_name, delimiter=sep, skip_header=0, max_rows=1, dtype=str)[1:]
    names_signatures = np.insert(names_signatures, 0, 'Samples', axis=0)
    signatures = np.delete(signatures, 0, axis=1)

    # Reorder signatures to standard mutation type order if appropriate
    try:
        mutation_types = get_mutation_types_from_file(file_name, format, sep)
        if len(mutation_types) == signatures.shape[0]:
            signatures = sort_matrix_rows(signatures, mutation_types)
    except Exception:
        pass

    return signatures, names_signatures

def create_folder_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as error:
        print(f"Błąd podczas tworzenia katalogu {folder_path}: {error}")

