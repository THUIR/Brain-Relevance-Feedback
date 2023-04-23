import json
import tqdm
import numpy as np
import pickle
import pandas as pd
from statsmodels.stats.anova import AnovaRM

def load_data():
    u2info = json.load(open('../release/u2info.json'))
    u2rel2eeg = {}
    for u in tqdm.tqdm(u2info.keys()):
        if u.startswith('0') or u.startswith('1'):
            if u not in u2rel2eeg.keys():
                u2rel2eeg[u] = {}
            rel2eeg = u2rel2eeg[u]
            eeg_data = json.load(open(f'../release/idx2eeg/{u}.json'))
            considered_idx = []
            for raw_q in u2info[u]['raw_q2raw_d2info'].keys():
                for item in u2info[u]['raw_q2raw_d2info'][raw_q]:
                    idx = str(item['idx'])
                    if idx in considered_idx:
                        continue
                    considered_idx.append(idx)

                    if len(np.array(eeg_data[idx]['raw']).shape) != 2 or np.array(eeg_data[idx]['raw']).shape[1] != 2000:
                        continue
                    rel = str(item['score'])
                    if rel not in rel2eeg.keys():
                        rel2eeg[rel] = {'raw':[], 'fs':[]}
                    # if np.max(np.array(eeg_data[idx]['raw'])[:,:1000]) - np.min(eeg_data[idx]['raw'][:1000]) < -100e-6:
                    rel2eeg[rel]['raw'].append(eeg_data[idx]['raw'][:1000])
                    rel2eeg[rel]['fs'].append(eeg_data[idx]['fs'][0])
    pickle.dump(u2rel2eeg, open('../release/u2rel2eeg.pkl', 'wb'))

def load_data2():
    u2info = json.load(open('../release/u2info.json'))
    u2rel2eeg = {}
    for u in tqdm.tqdm(u2info.keys()):
        if u.startswith('0') or u.startswith('1'):
            if u not in u2rel2eeg.keys():
                u2rel2eeg[u] = {}
            rel2eeg = u2rel2eeg[u]
            eeg_data = json.load(open(f'../release/idx2eeg/{u}.json'))
            considered_idx = []
            for raw_q in u2info[u]['raw_q2raw_d2info'].keys():
                for item in u2info[u]['raw_q2raw_d2info'][raw_q]:
                    idx = str(item['idx'])
                    if idx in considered_idx:
                        continue
                    considered_idx.append(idx)

                    if len(np.array(eeg_data[idx]['raw']).shape) != 2 or np.array(eeg_data[idx]['raw']).shape[1] < 2000:
                        continue
                    rel = str(item['score'])
                    if rel not in rel2eeg.keys():
                        rel2eeg[rel] = {'raw':[], 'fs':[]}

                    rel2eeg[rel]['fs'].append(eeg_data[idx]['fs'][0])
                    rel2eeg[rel]['fs'].append(eeg_data[idx]['fs'][1])

    pickle.dump(u2rel2eeg, open('../release/u2rel2eeg.pkl', 'wb'))

def process_data():
    u2rel2eeg = pickle.load(open('../release/u2rel2eeg.pkl', 'rb'))
    select_bands = [[0,180],[180,300],[300,500],[500,800]]
    for u in tqdm.tqdm(u2rel2eeg.keys()):
        for rel in u2rel2eeg[u].keys():
            tmp_dic = {'raw':np.zeros((62, 4)),'fs':np.zeros((62, 5))}
            for channel in range(62):
                selected_data = []
                for i in range(len(u2rel2eeg[u][rel]['fs'])):
                    selected_data.append(i)
                for band in range(5):
                    tmp_dic['fs'][channel][band]  = np.mean([u2rel2eeg[u][rel]['fs'][item][channel][band] for item in selected_data])
            u2rel2eeg[u][rel] = tmp_dic
    pickle.dump(u2rel2eeg, open('../release/u2rel2eeg.processed.pkl', 'wb'))

def process_data_merge():
    u2rel2eeg = pickle.load(open('../release/u2rel2eeg.pkl', 'rb'))
    select_bands = [[0,180],[180,300],[300,500],[500,800]]
    for u in u2rel2eeg.keys():
        for key in ['2','3']:
            if key in u2rel2eeg[u].keys():
                u2rel2eeg[u]['4']['raw'] += u2rel2eeg[u][key]['raw'] 
                u2rel2eeg[u]['4']['fs'] += u2rel2eeg[u][key]['fs']
                del u2rel2eeg[u][key]

    for u in tqdm.tqdm(u2rel2eeg.keys()):
        for rel in u2rel2eeg[u].keys():
            tmp_dic = {'raw':np.zeros((62, 4)),'fs':np.zeros((62, 5))}
            for channel in range(62):
                selected_data = []
                for i in range(len(u2rel2eeg[u][rel]['fs'])):
                    selected_data.append(i)
                for band in range(5):
                    tmp_dic['fs'][channel][band]  = np.mean([u2rel2eeg[u][rel]['fs'][item][channel][band] for item in selected_data])
            u2rel2eeg[u][rel] = tmp_dic
    pickle.dump(u2rel2eeg, open('../release/u2rel2eeg.processed.merged.pkl', 'wb'))


def compute_f():
    u2rel2eeg = pickle.load(open('../release/u2rel2eeg.processed.merged.pkl', 'rb'))

    value = np.zeros((62, 5))
    mask = np.zeros((62, 5))
    diff = np.zeros((62, 5))
    for channel in range(62):
        for band in range(5):
            df = pd.DataFrame(columns=['Treat', 'Value', 'd'])
            for u in u2rel2eeg.keys():
                if np.sum(np.isnan(u2rel2eeg[u]['1']['fs'][channel][band])) == 0 and np.sum(np.isnan(u2rel2eeg[u]['4']['fs'][channel][band])) == 0:
                    if '2' in u2rel2eeg[u].keys() and np.sum(np.isnan(u2rel2eeg[u]['2']['fs'][channel][band])) != 0:
                        u2rel2eeg[u]['2']['fs'][channel][band] = 0.75 * u2rel2eeg[u]['1']['fs'][channel][band] + 0.25 * u2rel2eeg[u]['4']['fs'][channel][band]
                    if '3' in u2rel2eeg[u].keys() and np.sum(np.isnan(u2rel2eeg[u]['3']['fs'][channel][band])) != 0:
                        u2rel2eeg[u]['3']['fs'][channel][band] = 0.25 * u2rel2eeg[u]['1']['fs'][channel][band] + 0.75 * u2rel2eeg[u]['4']['fs'][channel][band]
                    for rel in u2rel2eeg[u].keys(): # 
                        df.loc[len(df)] = [float(rel), u2rel2eeg[u][rel]['fs'][channel][band], u]

            anova = AnovaRM(df, 'Value', 'd', within=['Treat']).fit()
            value[channel][band] = anova.anova_table['F Value'][0]
            mask[channel][band] = anova.anova_table['Pr > F'][0]

    json.dump(value.tolist(), open('../release/fs_anova.json', 'w'))
    json.dump(mask.tolist(), open('../release/fs_mask.json', 'w'))

def paint_topo(rel_data, mask = None, name = ''):
    import mne
    import copy
    import matplotlib.pyplot as plt
    picked_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2", ]
    total_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2", ]
    rel_data = np.array(rel_data)

    if mask == None:
        mask = np.ones(rel_data.shape)

    mask_new = []
    for i in range(len(picked_channels)):
        if picked_channels[i] in total_channels:
            idx = total_channels.index(picked_channels[i])
            mask_new.append(mask[idx])

    data = rel_data

    montage = mne.channels.read_dig_fif('/home/yzy/zgg/152.136.32.8:8901/mode/montage.fif')
    montage.ch_names = json.load(open("/home/yzy/zgg/152.136.32.8:8901/mode/montage_ch_names.json"))
    montage.dig = montage.dig[:64]
    montage.ch_names = montage.ch_names[:64]
    for i in range(len(montage.dig)):
        montage.dig[i]['r'] = np.array([item * 1e-6 for item in montage.dig[i]['r']])

    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    for dig_info_ in ten_twenty_montage.dig:
        dig_info = copy.deepcopy(dig_info_)
        if 'EEG' not in str(dig_info['kind']):
            print(dig_info)
            montage.dig.insert(0, dig_info)

    fake_info = mne.create_info(ch_names=total_channels, sfreq=1000., ch_types='eeg')
    fake_evoked = mne.EvokedArray(data, fake_info).pick_channels(picked_channels)
    fake_evoked.set_montage(montage)

    data = fake_evoked.data
    data = np.array(data)
    fig, ax = plt.subplots(nrows = 1, ncols=6, figsize=(20, 10), gridspec_kw=dict(top=0.9), sharex=True, sharey=True)
    vmin = np.min(data)
    vmax = np.max(data)
    print(vmax, vmin)
    vmax = 30
    vmin = 0
    
    from matplotlib import colors 
    import matplotlib
    mycolor=['deepskyblue','skyblue','mediumspringgreen','yellow','orange','coral','lightcoral','red']
    cmap_color = colors.LinearSegmentedColormap.from_list('my_list', mycolor)

    for idx in range(rel_data.shape[1]):
        fs_data = data[:,idx]
        mask = np.array(mask_new)[:,idx] < 0.001
        mne.viz.plot_topomap(fs_data, fake_evoked.info, axes=ax[idx], show=False, vmin=vmin, vmax=vmax, mask = mask, mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=5),cmap=cmap_color)

    plt.savefig(f'../results/{name}.jpg')
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_color), ax=ax)
    plt.savefig(f'../results/colorbar.jpg')
    

if __name__ == '__main__':
    load_data2()
    process_data()
    process_data_merge()
    compute_f()
    erp_anova = json.load(open('../release/fs_anova.json'))
    erp_mask = json.load(open('../release/fs_mask.json'))
    paint_topo(erp_anova, erp_mask, name = 'fs_anova')



