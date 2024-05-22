
import os
import re
import torch
import pickle

import numpy as np
import os.path as osp
import torch.multiprocessing as mp

from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import FragmentCatalog, RDConfig, AllChem, MACCSkeys
from multiprocessing import Pool
from dataset.mol_features import allowable_features
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import InMemoryDataset, Data

RDLogger.DisableLog('rdApp.*')

class FewshotMolDataset(InMemoryDataset):
    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx, train_tasks, test_tasks]
    names = {
        'pcba': ['PCBA', 'pcba', 'pcba', -1, slice(0, 128), 118, 10],
        'muv': ['MUV', 'muv', 'muv', -1, slice(0, 17), 12, 5],
        'tox21': ['Tox21', 'tox21', 'tox21', -1, slice(0, 12), 9, 3],
        'sider': ['SIDER', 'sider', 'sider', 0, slice(1, 28), 21, 6],

        # toxcast subtask
        'toxcast-APR': ['ToxCast-APR', 'toxcast-APR', 'toxcast-APR', 0, slice(1, 44), 33, 10],
        'toxcast-ATG': ['ToxCast-ATG', 'toxcast-ATG', 'toxcast-ATG', 0, slice(1, 147), 106, 40],
        'toxcast-BSK': ['ToxCast-BSK', 'toxcast-BSK', 'toxcast-BSK', 0, slice(1, 116), 84, 31],
        'toxcast-CEETOX': ['ToxCast-CEETOX', 'toxcast-CEETOX', 'toxcast-CEETOX', 0, slice(1, 15), 10, 4],
        'toxcast-CLD': ['ToxCast-CLD', 'toxcast-CLD', 'toxcast-CLD', 0, slice(1, 20), 14, 5],
        'toxcast-NVS': ['ToxCast-NVS', 'toxcast-NVS', 'toxcast-NVS', 0, slice(1, 140), 100, 39],
        'toxcast-OT': ['ToxCast-OT', 'toxcast-OT', 'toxcast-OT', 0, slice(1, 16), 11, 4],
        'toxcast-TOX21': ['ToxCast-TOX21', 'toxcast-TOX21', 'toxcast-TOX21', 0, slice(1, 101), 80, 20],
        'toxcast-Tanguay': ['ToxCast-Tanguay', 'toxcast-Tanguay', 'toxcast-Tanguay', 0, slice(1, 19), 13, 5],
    }

    def __init__(self, root, name, workers, chunk_size, transform=None, pre_transform=None,
                 pre_filter=None, device=None):

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name
        self.device = device
        self.workers = workers
        self.chunk_size = chunk_size
        assert self.name in self.names.keys()
        super(FewshotMolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.n_task_train, self.n_task_test = self.names[self.name][5], self.names[self.name][6]
        self.total_tasks = self.n_task_train + self.n_task_test
        if name != 'pcba':
            self.train_task_range = list(range(self.n_task_train))
            self.test_task_range = list(range(self.n_task_train, self.n_task_train + self.n_task_test))
        else:
            self.train_task_range = list(range(5, self.total_tasks - 5))
            self.test_task_range = list(range(5)) + list(range(self.total_tasks - 5, self.total_tasks))

        self.data, self.slices = torch.load(self.processed_paths[0], map_location=device)
        self.index_list = pickle.load(open(self.processed_paths[1], 'rb'))
        self.y_matrix = np.load(open(self.processed_paths[2], 'rb'))

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt', 'index_list.pt', 'label_matrix.npz'

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        y_list = []
        data_id = 0
        smiles_list = []
        for line in tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')
            smiles = line[self.names[self.name][3]]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            Chem.Kekulize(mol)
            smiles_list.append(smiles)

            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1).to(self.device)
            y_list.append(ys)

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()))
                x.append(allowable_features['possible_chirality_list'].index(atom.GetChiralTag()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 2).to(self.device)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                e = []
                e.append(allowable_features['possible_bonds'].index(bond.GetBondType()))
                e.append(allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))
                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]
            edge_index = torch.tensor(edge_indices).to(self.device)
            edge_index = edge_index.t().to(torch.long).view(2, -1).to(self.device)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2).to(self.device)
            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles, id=data_id)
            data_id += 1
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        print('initial process finish')
        y_matrix = np.array(y_list)
        index_list = []  # [[[],[]], [[],[]]], task-label-index
        for task_i in range(y_matrix.shape[1]):
            task_i_label_values = y_matrix[:, task_i]
            class1_index = np.nonzero(task_i_label_values > 0.5)[0].tolist()
            class0_index = np.nonzero(task_i_label_values < 0.5)[0].tolist()
            index_list.append([class0_index, class1_index])

        pickle.dump(index_list, open(self.processed_paths[1], 'wb'))
        np.save(open(self.processed_paths[2], 'wb'), y_matrix)
        mp.set_sharing_strategy('file_system')
        finger_vec = None
        group_vec = None
        split_data = [smiles_list[i:i+self.chunk_size*self.workers] for i in range(0, len(smiles_list),
                                                                                   self.chunk_size*self.workers)]
        pool = Pool(self.workers)
        for i, data in enumerate(split_data):
            chunks = [(i, i + self.chunk_size, data[i:i + self.chunk_size]) if i + self.chunk_size < len(data) else (
                i, len(data), data[i:]) for i in range(0, len(data), self.chunk_size)]
            result_list = pool.map(compute_index, chunks)
            result_list = np.concatenate(result_list, 0)
            if finger_vec is None:
                finger_vec = result_list[:, 0:2214]
                group_vec = result_list[:, 2214:]
            else:
                finger_vec = np.vstack((finger_vec, result_list[:, 0:2214]))
                group_vec = np.vstack((group_vec, result_list[:, 2214:]))
        pool.close()
        pool.join()
        finger_vec = torch.FloatTensor(finger_vec).to(self.device)
        group_vec = torch.FloatTensor(group_vec).to(self.device)
        for i in tqdm(range(len(data_list))):
            data_list[i].__setattr__('fingerprint', finger_vec[i])
            data_list[i].__setattr__('groupprint', group_vec[i])
        print('extract finish!')
        torch.save(self.collate(data_list), self.processed_paths[0])


    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

def compute_index(args):

    start_idx, end_idx, data_list = args
    sim_chunk = []
    for i in tqdm(range(start_idx, end_idx), desc=f"Processing data_list ({start_idx} - {end_idx})"):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(data_list[i-start_idx])
        mol = Chem.MolFromSmiles(data_list[i-start_idx])
        groups_index = torch.zeros(49).bool()
        groups_index[Sample_Groups(mol)] = True
        fingerprint = torch.cat([torch.LongTensor(MACCSkeys.GenMACCSKeys(mol))[1:],
                                 torch.LongTensor(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))],
                                -1).bool().numpy() if scaffold != '' else torch.zeros(2214).bool().numpy()
        sim_chunk.append(np.hstack((fingerprint, groups_index.numpy())))
    return sim_chunk

def Sample_Groups(mol):
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    Groupids = []
    fcat = FragmentCatalog.FragCatalog(fparams)
    fcgen = FragmentCatalog.FragCatGenerator()
    fcgen.AddFragsFromMol(mol, fcat)
    num_entries = fcat.GetNumEntries()
    temp = []
    for i in range(0, num_entries):
        temp.extend(list(fcat.GetEntryFuncGroupIds(i)))
    Groupids.extend(set(temp))
    Groupids = list(set(Groupids))
    Groupids.sort()
    return Groupids


def scaffold_sim_compute(s_fingerprint, q_fingerprint):
    s_indices = torch.nonzero(s_fingerprint)
    q_indices = torch.nonzero(q_fingerprint)
    same_num = torch.concat([s_indices, q_indices]).shape[0] - torch.concat([s_indices, q_indices]).unique().shape[0]
    sca_sim = same_num * 2 / torch.concat([s_indices, q_indices]).shape[0]
    return sca_sim


def group_sim_compute(s_groups, q_groups):
    temp = s_groups.copy()
    temp.extend(q_groups)
    group_sim = 2 * (len(temp) - len(list(set(temp)))) / len(temp) if len(temp) != 0 else 0
    return group_sim