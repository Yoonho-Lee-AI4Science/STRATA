import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
import json
import random
import shutil
import copy
import torch
from functools import partial
from sklearn.model_selection import KFold, StratifiedKFold
import argparse
import pdb
import copy
import math
from sklearn.mixture import GaussianMixture

COMET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
TARGET_TRAIN_SIZE = 540
TARGET_VALID_SIZE = 180
TARGET_TEST_SIZE = 180
REQUIRED_SPLIT_SIZE = TARGET_TRAIN_SIZE + TARGET_VALID_SIZE + TARGET_TEST_SIZE
SELECTIVE_FILTER_CELLS = {"N2a", "ARPE19"}
SELECTIVE_TARGET_TRAIN_SIZE = 360
SELECTIVE_TARGET_VALID_SIZE = 120
SELECTIVE_TARGET_TEST_SIZE = 120
SELECTIVE_REQUIRED_SPLIT_SIZE = (
    SELECTIVE_TARGET_TRAIN_SIZE + SELECTIVE_TARGET_VALID_SIZE + SELECTIVE_TARGET_TEST_SIZE
)

SMILES_DLIN_MC3_DMA = 'CCCCC/C=C\\C/C=C\\CCCCCCCCC(CCCCCCCC/C=C\\C/C=C\\CCCCC)OC(=O)CCCN(C)C'
SMILES_CHOLESTEROL = 'C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C'
SMILES_DMG_PEG2000 = 'O=C(CCCCCCCCCCCCC)OCC(OC(CCCCCCCCCCCCC)=O)COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC'
SMILES_DOTAP = '[H]C(C[N+](C)(C)C)(OC(CCCCCCC/C=C\\CCCCCCCC)=O)COC(CCCCCCC/C=C\\CCCCCCCC)=O'
SMILES_DDAB = 'CCCCCCCCCCCCCCCCCC[N+](CCCCCCCCCCCCCCCCCC)(C)C'
SMILES_DOPE = '[H][C@@](COP([O-])(OCC[NH3+])=O)(OC(CCCCCCC/C=C\\CCCCCCCC)=O)COC(CCCCCCC/C=C\\CCCCCCCC)=O'
SMILES_DSPC = 'CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCCC'
SMILES_14PA = '[P](=O)([O-])(OC[C@H](OC(=O)CCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCC)O'
SMILES_18PG = '[P](=O)([O-])(OC[C@H](OC(=O)CCCCCCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCCCCCC)OCC(O)CO'

HELPER_LIPID_TO_SMILES = {
    'DOTAP': SMILES_DOTAP,
    'DDAB': SMILES_DDAB,
    'DOPE': SMILES_DOPE,
    'DSPC': SMILES_DSPC,
    '14PA': SMILES_14PA,
    '18PG': SMILES_18PG,
}



def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def get_default_split_sizes(args, dataset_sz):
    if args.cell in SELECTIVE_FILTER_CELLS:
        required_split_size = SELECTIVE_REQUIRED_SPLIT_SIZE
        target_train_size = SELECTIVE_TARGET_TRAIN_SIZE
        target_valid_size = SELECTIVE_TARGET_VALID_SIZE
        target_test_size = SELECTIVE_TARGET_TEST_SIZE
    else:
        required_split_size = REQUIRED_SPLIT_SIZE
        target_train_size = TARGET_TRAIN_SIZE
        target_valid_size = TARGET_VALID_SIZE
        target_test_size = TARGET_TEST_SIZE

    if dataset_sz < required_split_size:
        raise ValueError(
            f"Dataset size {dataset_sz} is smaller than required default split size {required_split_size}."
        )
    return target_train_size, target_valid_size, target_test_size


def has_valid_label(value):
    if pd.isna(value):
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return False
    return True

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    coordinates = smi2_2Dcoords(smi) 
        except:
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

def inner_lnp2data(smi2mol_id, content, pickle_output=True):
    components_list = content['components']
    if "labels" in content:
        raw_labels = content['labels']
    else:
        raw_labels = {}
    dataset_name = content['dataset_name']
    lnp_id = content['lnp_id']

    # handle non-core (optional) attributes of LNPs, e.g. NP_ratio
    np_props = {}
    if "NP_ratio" in content:
        np_props['NP_ratio'] = content['NP_ratio']
    if "actual_ilrna_wt_ratio" in content:
        np_props['actual_ilrna_wt_ratio'] = content['actual_ilrna_wt_ratio']
    if "volumetric_ratio" in content:
        np_props['volumetric_ratio'] = content['volumetric_ratio']        

    labels = raw_labels # raw_labels is already normalized

    output_components_list = []
    
    mol_ids = []
    percents = []
    component_types = []
    # reaction_steps = []

    for component in components_list:

        # component_output = component.copy()
        component_output = copy.deepcopy(component)
        mol_id = smi2mol_id[component['smi']]
        component_output['mol_id'] = mol_id

        output_components_list.append(component_output)

        mol_ids.append(mol_id)

        percent = component['percent']
        percents.append(percent)

        component_type = component['component_type']
        component_types.append(component_type)


    output = {
        'mol_id': mol_ids, 
        'percent': percents, 'component_type': component_types, 
        'target': labels, 
        'dataset_name': dataset_name, 
        'components': output_components_list,
        'lnp_id': lnp_id,
        **np_props # fold non-core (optional) attributes of LNPs into sample dict, e.g. NP_ratio, volumetric_ratio
        }

    # print("inner_lnp2data output: ", output)
    
    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        output

def lnp2data(smi2mol_id, content):
    try:
        return inner_lnp2data(smi2mol_id, content)
    except:
        print("failed lnp: {}".format(content[0]))
        return None

def inner_smi2coords(content, pickle_output=False):

    smi = content

    cnt = 10 # conformer num,all==11, 10 3d + 1 2d
    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 

    output = {'atoms': atoms, 
    'coordinates': coordinate_list, 
    'mol': mol,'smi': smi}
    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        return output
        
def smi2coords_onlymol(content):
    try:
        return inner_smi2coords_onlymol(content)
    except:
        print("failed smiles: {}".format(content))
        return None

def inner_smi2coords_onlymol(content, pickle_output=True):
    output = inner_smi2coords(content)
    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        output

# with fixnoutputlnp_ids : to generate dataset based on given lnp_ids and output lnp_ids in train, valid and test sets
# and splitlabeldata
def create_lmdb_and_sampling(args = None, nthreads=16, debug=False, shuffle=True):
    assert args != None
    dataset_name = f'LC24{args.cell}_form_screen'
    label_column = f'RLU_{args.cell}'
    inpath=COMET_DIR+'experiments/data_json/LNP_selective.csv'
    outpath=COMET_DIR+f'experiments/processed_data_dirs/{dataset_name}'
    if args.unseen_ILs:
        outpath += '_usIL'
    elif args.unseen_HLs:
        outpath += '_usHL'
    elif args.unseen_split_ILs:
        outpath += '_usILratio'
    elif args.unseen_split_HLs:
        outpath += '_usHLratio'
    elif args.unseen_split_CHOLs:
        outpath += '_usCHOLratio'
    elif args.unseen_split_PEGs:
        outpath += '_usPEGratio'
    elif args.unseen_split_CARs:
        outpath += '_usCARratio'
    else: 
        outpath += ''
        
    if not os.path.exists(outpath):
        os.makedirs(outpath)


    # function creates a test set made up of top and bottom subset of the dataset, if not defined, function create a test set randomly (with size of test_ratio) 
    #total_random_test_ratio = test_ratio - top_bottom_ratio
    #remaining_random_test_ratio = total_random_test_ratio / (1 - top_bottom_ratio) # find the ratio of remaining dataset to randomly sample for test set after taking out top and bottom heldout set

    # Data wil be stored in JSON format, e.g. each sample: {components: [{smi: <SMILES>, percent: <%>, name: IL-1}, {..}], label: <label_value>}
    

    #DOTAP_SMILES = 'CCCCCCCC/C=C\CCCCCCCC(=O)OCC(C[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC' # double check
    #DOPE_SMILES  = 'CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCC/C=C\CCCCCCCC' # double check

    #DSPC_SMILES  = '[C@](COP(=O)([O-])OCC[N+](C)(C)C)([H])(OC(CCCCCCCCCCCCCCCCC)=O)COC(CCCCCCCCCCCCCCCCC)=O' # double check # from LIPIDMAPS 

    #CHOL_SMILES = 'C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C'

    #PEG_SMILES = '[H][C@@](COP([O-])(OCCNC(OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC)=O)=O)(OC(CCCCCCCCCCCCC)=O)COC(CCCCCCCCCCCCC)=O.[NH4+]'
    #HL_SMILES = {'DOTAP':DOTAP_SMILES, 'DOPE':DOPE_SMILES, 'DSPC':DSPC_SMILES}
    
    read_file_csv = pd.read_csv(inpath)
    if label_column not in read_file_csv.columns:
        raise ValueError(f"Column {label_column} not found in {inpath}")

    rows = []
    for lnp_idx, csv_row in read_file_csv.iterrows():
        label_value = csv_row[label_column]
        if args.cell in SELECTIVE_FILTER_CELLS and not has_valid_label(label_value):
            continue

        helper_lipid = str(csv_row['Helper_lipid']).strip()
        if helper_lipid not in HELPER_LIPID_TO_SMILES:
            raise ValueError(f"Unsupported helper lipid: {helper_lipid}")

        il_hl_total = float(csv_row['(IL+HL)'])
        hl_ratio = il_hl_total * float(csv_row['HL_(IL+HL)']) / 100
        il_ratio = il_hl_total - hl_ratio
        peg_chol_total = 100 - il_hl_total
        peg_ratio = peg_chol_total * float(csv_row['PEG_(Chol+PEG)']) / 100
        chol_ratio = peg_chol_total - peg_ratio

        rows.append([
            lnp_idx,
            str(label_value),
            SMILES_DLIN_MC3_DMA,
            HELPER_LIPID_TO_SMILES[helper_lipid],
            SMILES_CHOLESTEROL,
            SMILES_DMG_PEG2000,
            helper_lipid,
            str(csv_row['NP_ratio']),
            str(il_ratio),
            str(hl_ratio),
            str(chol_ratio),
            str(peg_ratio),
            dataset_name,
        ])

    column_names = ['lnp_id', 'label', 'IL_SMILES', 'HL_SMILES', 'CHOL_SMILES', 'PEG_SMILES', 'HL_name', 'mRNA_weight', 'IL_ratio', 'HL_ratio', 'CHOL_ratio', 'PEG_ratio', 'dataset_name']

    
    json_obj = []
    for row_idx, row in enumerate(rows):
        components = [{"smi":row[2], "component_type":"IL", "mol":float(row[8])/100, "percent":float(row[8])/100}, \
            {"smi":row[3], "component_type":"HL", "mol":float(row[9])/100, "percent":float(row[9])/100}, \
            {"smi":row[4], "component_type":"CH", "mol":float(row[10])/100, "percent":float(row[10])/100}, \
            {"smi":row[5], "component_type":"PEG", "mol":float(row[11])/100, "percent":float(row[11])/100}
        ]
        labels = {"{}_label".format(row[-1]):float(row[1])}
        new_lnp = {"components": components, \
                    "labels":labels, \
                    "volumetric_ratio": "3:1", \
                    "il_rna_wt_ratio": "{}:1".format(row[7]), \
                    "dataset_name": row[-1], \
                    "phase": "3", \
                    "lipid_ratio": "I1", \
                    "actual_ilrna_wt_ratio": float(row[7]), \
                    "NP_ratio": float(row[7]), \
                    "lnp_id": str(row[0]) \
                    }
        json_obj.append(new_lnp)
    # hl_list = list(set([aa[2] for aa in column_selected_csv[1:]]))

    
    dataset_name_list = []
    dataset_dict = {}

    # collate into list of lnps
    
    
    json_list = []
    for lnp_dict in json_obj:
        #lnp_dict = json_obj[lnp_id]
        #lnp_dict['lnp_id'] = lnp_id # Give LNP its id number
        #if not args.allow_multiple_ILs:
        #    if [com['component_type'] for com in lnp_dict['components']].count('IL') > 1:
        #        continue
            
        
        if 'dataset_name' in lnp_dict: # lnp_dict['dataset_name'] --> 'A549_form_screen'
            lnp_dataset_name = lnp_dict['dataset_name']
            if lnp_dataset_name not in dataset_name_list:
                dataset_name_list.append(lnp_dataset_name)
                dataset_dict[lnp_dataset_name] = []
            dataset_dict[lnp_dataset_name].append(lnp_dict)
        # process components' percent value
        np_components = lnp_dict['components'] # {{smi, component_type, mol} for four components(i.e. IL, HL, CH, PEG)}
        # total_weight = 0

        # record percent composition in component dict
        for c_id, component in enumerate(np_components):
            lnp_dict['components'][c_id]['percent'] = lnp_dict['components'][c_id]['mol'] # paraphrase 'mol' to 'percent'
        json_list.append(lnp_dict)
        
    # reindexing lnp_id
    #sz = len(json_list)
    #sorted_lnp_id = list(map(str,sorted([int(lnp['lnp_id']) for lnp in json_list])))
    #for idx in range(sz):
    #    json_list[idx]['lnp_id'] = str(sorted_lnp_id.index(json_list[idx]['lnp_id']))
        

    

    # make master mol lmdb dataset as a list of all unique mols in datasets
    smi_list = []
    for sample_i, np_obj in enumerate(json_list):
        np_components = np_obj['components']
        unique_smi_count = 0
        for component in np_components:
            smi = component['smi']
            if smi not in smi_list:
                unique_smi_count += 1
                smi_list.append(smi) 
    # smi_list contains non-redundant smiles list

    mol_filename = "mol.lmdb"
    os.makedirs(outpath, exist_ok=True)
    mol_output_name = os.path.join(outpath, mol_filename) # lance_default/mol.lmdb
    if os.path.isfile(mol_output_name):
        print("finished processing mol.lmdb")
    else:
        try:
            os.remove(mol_output_name)
        except:
            pass
        env_new = lmdb.open( mol_output_name, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, max_readers=1, map_size=int(100e9), )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(smi2coords_onlymol, smi_list)):
                if inner_output is not None:
                    print("i=", i, " data = pickle.loads(datapoint_pickled) smi: ", pickle.loads(inner_output)['smi'])
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
            print('{} process {} lines'.format(mol_filename, i))
            txn_write.commit()
            env_new.close()
        print("finished processing mol.lmdb")
    
    

    # make smi2mol_id dict
    smi2mol_id = {}
    mol_id2smi = {}

    # for ind, lnp_id in enumerate(json_obj):
    for mol_id, smi in enumerate(smi_list):
        smi2mol_id[smi] = mol_id
        mol_id2smi[mol_id] = smi
    # lmdb smiles index mapping created.
    

    lnp2data_w_smi2mol_id = partial(lnp2data, smi2mol_id) # smi2mol_id is the dict to map smi to mol_id in mol.lmdb

    # use index of smi_list as pointer in NP data split and as key to access mol data in mol.lmdb
    # each row in train.lmdb correspond to a NP sample, with its a) components' i) mol_id, ii) percent and b) label
    


    def make_data_lmdb(args, seed, train, valid, test, dataset_outpath, debug=False):

        mol_file_name = dataset_outpath + '/mol.lmdb'
        dataset_outpath += f'/{args.train_ratio}_{args.val_ratio}_{args.test_ratio}/split_{seed}/'
        os.makedirs(dataset_outpath, exist_ok=True)
        shutil.copyfile(mol_file_name, dataset_outpath+'mol.lmdb')
        dataset_outpath += f'{dataset_name}/'
        os.makedirs(dataset_outpath, exist_ok=True)
        
        
        for name, content_list in [('train.lmdb', train), ('valid.lmdb', valid), ('test.lmdb', test)]:

            
            if debug:
                output_json_name = os.path.join(dataset_outpath, name.replace(".lmdb", ".json"))
                #json_object = json.dumps(content_list, indent=4, default= set_default)
                json_object = json.dumps(content_list, indent=4)
                with open(output_json_name, "w") as outfile:
                    outfile.write(json_object)
            
            output_name = os.path.join(dataset_outpath, name)

            try:
                os.remove(output_name)
            except:
                pass

            
            env_new = lmdb.open( output_name, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, max_readers=1, map_size=int(100e9), )
            txn_write = env_new.begin(write=True)
            with Pool(nthreads) as pool:
                i = 0
                for inner_output in tqdm(pool.imap(lnp2data_w_smi2mol_id, content_list)):
                    if inner_output is not None:
                        txn_write.put(f'{i}'.encode("ascii"), inner_output)
                        i += 1
                print('{} process {} lines'.format(name, i))
                txn_write.commit()
                env_new.close()
        return dataset_outpath
                

    for row_idx, row in enumerate(rows):
        rows[row_idx] = [str(row[0])] + row[1:]
    seed_list = [i for i in range(20)]
    for seed in seed_list:
        dataset_json_list = copy.deepcopy(dataset_dict[dataset_name])
        dataset_sz = len(dataset_json_list)
        random.seed(seed)
        np.random.seed(seed)
        # Shuffle json_obj
        # TODO NOW: shuffle dataset indices here to get different folds for train/valid/test splits
        if args.unseen_ILs:
            train, valid, test = handle_unseen_IL(args, dataset_json_list, column_names, dataset_sz)
        elif args.unseen_split_ILs:
            train, valid, test = handle_unseen_split(args, dataset_json_list, column_names, dataset_sz, 'IL')
        elif args.unseen_split_HLs:
            train, valid, test = handle_unseen_split(args, dataset_json_list, column_names, dataset_sz, 'HL')
        elif args.unseen_split_CHOLs:
            train, valid, test = handle_unseen_split(args, dataset_json_list, column_names, dataset_sz, 'CH')
        elif args.unseen_split_PEGs:
            train, valid, test = handle_unseen_split(args, dataset_json_list, column_names, dataset_sz, 'PEG')
        elif args.unseen_split_CARs:
            train, valid, test = handle_unseen_split_car(args, dataset_json_list, column_names, dataset_sz)
        elif args.unseen_HLs:
            train, valid, test = handle_unseen_HL(args, dataset_json_list, column_names, dataset_sz)
        else:
            np.random.shuffle(dataset_json_list)
            target_train_size, target_valid_size, target_test_size = get_default_split_sizes(args, dataset_sz)
            train = dataset_json_list[:target_train_size]
            valid = dataset_json_list[target_train_size:target_train_size + target_valid_size]
            test = dataset_json_list[target_train_size + target_valid_size:target_train_size + target_valid_size + target_test_size]
        train_idx = sorted([int(aa['lnp_id']) for aa in train])
        valid_idx = sorted([int(aa['lnp_id']) for aa in valid])
        test_idx = sorted([int(aa['lnp_id']) for aa in test])
        #train = subsample_train(train, random_train_subsample_ratio, train_subsample_sample_ids, subsample_target_label)

        
        split_dir = make_data_lmdb(args, seed, train, valid, test, outpath, debug)
        export_to_tsv(args, split_dir, rows, column_names, train_idx, valid_idx, test_idx)
        
    return 

def handle_unseen_IL(args, dataset_json_list, column_names, dataset_sz):
    il_smiles_list = [bb['smi']  for aa in dataset_json_list for bb in aa['components'] if bb['component_type'] == 'IL']
    # let's count
    count_smiles = dict()
    for smiles in il_smiles_list:
        if smiles not in count_smiles.keys():
            count_smiles[smiles] = 1
        else: 
            count_smiles[smiles] += 1
            
    # let's shuffle
    keys = list(count_smiles.keys())
    np.random.shuffle(keys)
    values = [count_smiles[key] for key in keys]
    train_iter_idx = 0
    train_sum = 0
    test_iter_idx = 0
    test_sum = 0
    for iter_idx in range(len(values)):
        train_sum += values[iter_idx]
        if train_sum >= (dataset_sz *args.train_ratio):
            train_iter_idx = iter_idx
            break
    for iter_idx in range(len(values),train_iter_idx+1, -1):
        test_sum += values[iter_idx-1]
        if test_sum >= (dataset_sz *args.test_ratio):
            test_iter_idx = iter_idx+1
            break
    
    train_keys = keys[:train_iter_idx+1]
    valid_keys = keys[train_iter_idx+1:test_iter_idx-1]
    test_keys = keys[test_iter_idx-1:]
    
    train = []
    valid = []
    test = []
    for final_iter_idx, lnp in enumerate(dataset_json_list):
        for component_idx, component in enumerate(lnp['components']):
            if component["component_type"] == 'IL':
                smiles = component["smi"]
                break
        if smiles in train_keys:
            train.append(lnp)
        elif smiles in valid_keys:
            valid.append(lnp)
        elif smiles in test_keys:
            test.append(lnp)
        else: 
            raise NotImplementedError
        
    return train, valid, test


def handle_unseen_HL(args, dataset_json_list, column_names, dataset_sz):
    target_train_size, target_valid_size, target_test_size = get_default_split_sizes(args, dataset_sz)
    hl_to_samples = {}
    for lnp in dataset_json_list:
        hl_smiles = None
        for component in lnp['components']:
            if component["component_type"] == 'HL':
                hl_smiles = component["smi"]
                break
        if hl_smiles is None:
            raise ValueError(f"HL component not found for lnp_id={lnp['lnp_id']}")
        if hl_smiles not in hl_to_samples:
            hl_to_samples[hl_smiles] = []
        hl_to_samples[hl_smiles].append(lnp)

    hl_keys = list(hl_to_samples.keys())
    if len(hl_keys) < 3:
        raise ValueError("At least three HL species are required to create train/valid/test splits.")

    np.random.shuffle(hl_keys)
    candidate_hls = [hl for hl in hl_keys if len(hl_to_samples[hl]) >= target_valid_size]
    if len(candidate_hls) < 2:
        raise ValueError(
            f"Need at least two HL species with >= {target_valid_size} samples for unseen_HL split."
        )

    valid_hl = candidate_hls[0]
    test_hl = candidate_hls[1]

    valid_source = list(hl_to_samples[valid_hl])
    test_source = list(hl_to_samples[test_hl])
    valid_selected_idx = np.random.choice(len(valid_source), size=target_valid_size, replace=False)
    test_selected_idx = np.random.choice(len(test_source), size=target_test_size, replace=False)
    valid = [valid_source[idx] for idx in valid_selected_idx]
    test = [test_source[idx] for idx in test_selected_idx]

    remaining_samples = []
    for hl_smiles in hl_keys:
        if hl_smiles in {valid_hl, test_hl}:
            continue
        remaining_samples.extend(hl_to_samples[hl_smiles])

    if target_train_size > len(remaining_samples):
        raise ValueError(
            f"Not enough remaining samples to draw {target_train_size} train samples "
            f"after reserving valid/test HLs."
        )

    selected_train_idx = np.random.choice(len(remaining_samples), size=target_train_size, replace=False)
    train = [remaining_samples[idx] for idx in selected_train_idx]

    return train, valid, test

def gmm_1d_auto_k(
        X,
        k_max: int = 20,
        criterion: str = "bic",   # "bic" or "aic"
        covariance_type: str = "full",
        n_init: int = 10,
        random_state: int = 42,
    ):

    scores = {}
    models = {}

    for k in range(1, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=random_state,
        )
        gmm.fit(X)

        score = gmm.bic(X) 
        scores[k] = float(score)
        models[k] = gmm

    best_k = min(scores, key=scores.get)
    best_model = models[best_k]

    labels = best_model.predict(X)
    probs = best_model.predict_proba(X)

    cov = best_model.covariances_
    variances = cov.reshape(-1)

    means = best_model.means_.reshape(-1)
    weights = best_model.weights_.reshape(-1)

    order = np.argsort(means)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(best_k)

    means = means[order]
    variances = variances[order]
    weights = weights[order]
    probs = probs[:, order]
    labels = inv_order[labels]

    return {
        "best_k": best_k,
        "best_model": best_model,
        "scores": scores,
        "labels": labels,
        "probs": probs,
        "means": means,
        "variances": variances,
        "weights": weights,
        "criterion": criterion.lower(),
    }

def handle_unseen_split(args, dataset_json_list, column_names, dataset_sz, comp_type ='IL'):
    target_train_size, target_valid_size, target_test_size = get_default_split_sizes(args, dataset_sz)
    ratio_list = np.asarray([bb['percent']  for aa in dataset_json_list for bb in aa['components'] if bb['component_type'] == comp_type], dtype=float).reshape(-1,1)
    lnp_id_list = np.asarray([aa['lnp_id']  for aa in dataset_json_list], dtype=str).reshape(-1,1)
    sorted_idx = ratio_list.squeeze(-1).argsort()
    num_sample = ratio_list.shape[0]
    num_val = target_valid_size
    num_test = target_test_size
    num_train = target_train_size

    val_start_max = num_sample - num_val
    val_start = np.random.randint(0, val_start_max + 1)
    val_idx = sorted_idx[val_start:val_start + num_val]

    remain_sorted_idx = sorted_idx[np.isin(sorted_idx, val_idx, invert=True)]
    remain_size = remain_sorted_idx.shape[0]

    test_start_max = remain_size - num_test
    test_start = np.random.randint(0, test_start_max + 1) if num_test > 0 else 0
    test_idx = remain_sorted_idx[test_start:test_start + num_test]

    candidate_train_idx = remain_sorted_idx[np.isin(remain_sorted_idx, test_idx, invert=True)]
    if num_train > candidate_train_idx.shape[0]:
        raise ValueError(
            f"Not enough remaining samples to draw {num_train} train samples after valid/test selection."
        )
    selected_train_positions = np.random.choice(candidate_train_idx.shape[0], size=num_train, replace=False)
    train_idx = candidate_train_idx[selected_train_positions]
    train_lnp_id = set(lnp_id_list[train_idx].squeeze(-1))
    valid_lnp_id = set(lnp_id_list[val_idx].squeeze(-1))
    test_lnp_id = set(lnp_id_list[test_idx].squeeze(-1))
    
    train =[]
    valid = []
    test = []
    for final_iter_idx, lnp in enumerate(dataset_json_list):
        lnp_id = lnp['lnp_id']
        if lnp_id in train_lnp_id:
            train.append(lnp)
        elif lnp_id in valid_lnp_id:
            valid.append(lnp)
        elif lnp_id in test_lnp_id:
            test.append(lnp)
    return train, valid, test

def handle_unseen_split_car(args, dataset_json_list, column_names, dataset_sz):
    if args.cell in SELECTIVE_FILTER_CELLS:
        target_split_size = 240
    else:
        target_split_size = 360
    ratio_to_samples = {}
    for lnp in dataset_json_list:
        ratio = float(lnp['NP_ratio'])
        if ratio not in ratio_to_samples:
            ratio_to_samples[ratio] = []
        ratio_to_samples[ratio].append(lnp)

    ratio_keys = sorted(ratio_to_samples.keys())
    if len(ratio_keys) != 3:
        raise ValueError(f"CAR split expects exactly 3 unique cargo ratios, found {len(ratio_keys)}: {ratio_keys}")

    for ratio in ratio_keys:
        if len(ratio_to_samples[ratio]) != target_split_size:
            raise ValueError(
                f"CAR split expects {target_split_size} samples for ratio {ratio}, found {len(ratio_to_samples[ratio])}"
            )

    shuffled_ratio_keys = ratio_keys.copy()
    np.random.shuffle(shuffled_ratio_keys)

    train_ratio, valid_ratio, test_ratio = shuffled_ratio_keys
    train = ratio_to_samples[train_ratio]
    valid = ratio_to_samples[valid_ratio]
    test = ratio_to_samples[test_ratio]

    if len(train) != target_split_size or len(valid) != target_split_size or len(test) != target_split_size:
        raise ValueError(
            f"Unexpected split sizes for CAR split: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )
    return train, valid, test

def export_to_tsv(args, split_dir, rows, column_names, train_idx, valid_idx, test_idx):
    text = ['\t'.join(row) for row in rows]
    column_text = '\t'.join(column_names) + '\n'
    train_text = column_text + '\n'.join([text[train_curr_idx] for train_curr_idx in train_idx]) +'\n'
    valid_text = column_text + '\n'.join([text[valid_curr_idx] for valid_curr_idx in valid_idx]) +'\n'
    test_text = column_text + '\n'.join([text[test_curr_idx] for test_curr_idx in test_idx]) +'\n'
    
    train_open = open('{}train.tsv'.format(split_dir), 'w')
    train_open.write(train_text)
    train_open.close()
    valid_open = open('{}valid.tsv'.format(split_dir), 'w')
    valid_open.write(valid_text)
    valid_open.close()
    test_open = open('{}test.tsv'.format(split_dir), 'w')
    test_open.write(test_text)
    test_open.close()
    return


def parse_args():
    
    parser = argparse.ArgumentParser(description="implicit_relational_GNN")
    parser.add_argument('--train_ratio', type=float, default = 60) 
    parser.add_argument('--val_ratio', type=float, default = 20) 
    parser.add_argument('--test_ratio', type=float, default = 20) 
    parser.add_argument('--cell', type=str, default = 'N2a') 
    #parser.add_argument('--allow_multiple_ILs', action='store_true') 
    parser.add_argument('--unseen_ILs', action='store_true') 
    parser.add_argument('--unseen_HLs', action='store_true')
    parser.add_argument('--unseen_split_ILs'  , action='store_true') 
    parser.add_argument('--unseen_split_HLs'  , action='store_true') 
    parser.add_argument('--unseen_split_CHOLs', action='store_true') 
    parser.add_argument('--unseen_split_PEGs' , action='store_true') 
    parser.add_argument('--unseen_split_CARs' , action='store_true') 
    args = parser.parse_args()
    
    args.train_ratio /= 100
    args.val_ratio /= 100
    args.test_ratio /= 100
    
    return args




def main():
    # random seed
    args = parse_args()
    create_lmdb_and_sampling( args=args, nthreads=8, debug=True)
    return







if __name__ == "__main__":
    main()
