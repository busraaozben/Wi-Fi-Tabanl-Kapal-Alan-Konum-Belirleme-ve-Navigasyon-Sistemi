#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KNN Modeli
WiFi RSSI parmak izi verisiyle KNN tabanlı mekan konumlandrma modeli
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# --- Dosya yollar ---
CSV_PATH = Path(r"D:/2026-2027_Guz_Donemi/bilgisayar_muhendisligi_uygulamalari/06 ekim/zemin_kat_veri_seti.csv")
OUTPUT_DIR = Path(r"D:/2026-2027_Guz_Donemi/bilgisayar_muhendisligi_uygulamalari/06 ekim/knn_model_sonuclari")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  NEÜ IPS Projesi - KNN Model Eğitimi")
print("=" * 60)


print("\n Veri okunuyor...")
df = pd.read_csv(CSV_PATH, sep=None, engine='python')


rp_coords = df.drop_duplicates('reference_point')[
    ['reference_point', 'location', 'pixel_x', 'pixel_y']
].set_index('reference_point')

fingerprint = df.pivot_table(
    values='rssi',
    index='reference_point',
    columns='bssid',
    aggfunc='mean'
)

fingerprint = fingerprint.fillna(-100)

rp_order = sorted(fingerprint.index, key=lambda x: int(x.replace('RP', '')))
fingerprint = fingerprint.reindex(rp_order)

print(f"\n{'='*60}")
print(f"  1. PARMAK Z MATRS")
print(f"{'='*60}")
print(f"   Boyut: {fingerprint.shape[0]} RP x {fingerprint.shape[1]} AP")
print(f"\n   Parmak izi matrisi (ilk 5 AP):")
print(fingerprint.iloc[:, :5].round(1).to_string())

X = fingerprint.values
y = fingerprint.index.values

print(f"\n   X (Özellik matrisi) : {X.shape}")
print(f"   y (etiketler)       : {y.shape}")

print(f"\n{'='*60}")
print(f"  2. VERİ ARTIRMA (Data Augmentation)")
print(f"{'='*60}")

np.random.seed(42)
N_AUGMENT = 10
NOISE_STD = 3

X_aug = []
y_aug = []

for i in range(len(X)):
    X_aug.append(X[i])
    y_aug.append(y[i])

    for _ in range(N_AUGMENT):
        noise = np.random.normal(0, NOISE_STD, size=X[i].shape)
        noisy_sample = X[i] + noise
        noisy_sample = np.clip(noisy_sample, -100, -20)
        X_aug.append(noisy_sample)
        y_aug.append(y[i])

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

print(f"   Orijinal veri  : {X.shape[0]} örnek")
print(f"   Arttırılmış veri: {X_aug.shape[0]} örnek ({N_AUGMENT} kopya/RP, {NOISE_STD} dBm gürültü)")

print(f"\n{'='*60}")
print(f"  3. ETM / TEST AYIRMA")
print(f"{'='*60}")

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(sss.split(X_aug, y_aug))

X_train = X_aug[train_idx]
y_train = y_aug[train_idx]
X_test = X_aug[test_idx]
y_test = y_aug[test_idx]

print(f"   Eğitim seti : {X_train.shape[0]} örnek")
print(f"   Test seti   : {X_test.shape[0]} örnek")
print(f"   Test oran  : %{len(X_test)/len(X_aug)*100:.0f}")


print(f"\n{'='*60}")
print(f"  4. OPTİMAL K DEĞERİ ARAMA")
print(f"{'='*60}")

k_range = range(1, 16)
k_results = {}

print(f"\n   {'K':<5} {'Eğitim %':<12} {'Test %':<12} {'Durum'}")
print(f"   {'-'*5} {'-'*12} {'-'*12} {'-'*15}")

best_k = 1
best_test_acc = 0

for k in k_range:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',
        metric='euclidean' #cityblock ya da manhattan
    )
    knn.fit(X_train, y_train)
    
    train_acc = knn.score(X_train, y_train) * 100
    test_acc = knn.score(X_test, y_test) * 100
    
    k_results[k] = {'train': train_acc, 'test': test_acc}
    
    marker = ""
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_k = k
        marker = "  EN İYİ"
    
    print(f"   K={k:<3} {train_acc:<12.1f} {test_acc:<12.1f} {marker}")

print(f"\n    Optimal K = {best_k} (Test doğruluğu: %{best_test_acc:.1f})")


print(f"\n{'='*60}")
print(f"  5. FİNAL MODEL (K={best_k})")
print(f"{'='*60}")

test_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
test_knn.fit(X_train, y_train)
y_pred = test_knn.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Test doğruluğu: %{test_accuracy*100:.1f}")
print(f"   Doğru tahmin  : {sum(y_test == y_pred)}/{len(y_test)}")

yanlis = y_test != y_pred
if yanlis.any():
    print(f"\n    Yanlış tahminler ({sum(yanlis)} adet):")
    print(f"   {'Gerçek':<10} {'Tahmin':<10} {'Gerçek Konum':<25} {'Tahmin Konum'}")
    print(f"   {'-'*10} {'-'*10} {'-'*25} {'-'*25}")
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            g_loc = rp_coords.loc[y_test[i], 'location']
            t_loc = rp_coords.loc[y_pred[i], 'location']
            print(f"   {y_test[i]:<10} {y_pred[i]:<10} {g_loc:<25} {t_loc}")
else:
    print(f"\n    Tüm tahminler doğru!")


final_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
final_knn.fit(X, y)


y_self_pred = final_knn.predict(X)
self_acc = accuracy_score(y, y_self_pred)

print(f"\n   Final model (orijinal veriyle):")
print(f"   Self-test doğruluğu: %{self_acc*100:.1f}")


print(f"\n   {'RP':<8} {'Konum':<25} {'Tahmin':<8} {'Sonuç'}")
print(f"   {'-'*8} {'-'*25} {'-'*8} {'-'*6}")
for i in range(len(y)):
    loc = rp_coords.loc[y[i], 'location']
    sonuc = "" if y[i] == y_self_pred[i] else f"{y_self_pred[i]}"
    print(f"   {y[i]:<8} {loc:<25} {y_self_pred[i]:<8} {sonuc}")


print(f"\n{'='*60}")
print(f"   Grafikler oluşturuluyor...")
print(f"{'='*60}")


fig, ax = plt.subplots(figsize=(10, 5))
ks = list(k_results.keys())
train_accs = [k_results[k]['train'] for k in ks]
test_accs = [k_results[k]['test'] for k in ks]

ax.plot(ks, train_accs, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Eğitim')
ax.plot(ks, test_accs, 's-', color='#FF5722', linewidth=2, markersize=8, label='Test')
ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.7, label=f'Optimal K={best_k}')

for k, acc in zip(ks, test_accs):
    ax.annotate(f'%{acc:.0f}', (k, acc), textcoords="offset points", 
                xytext=(0, -15), ha='center', fontsize=8, color='#FF5722')

ax.set_xlabel('K Değeri', fontsize=12)
ax.set_ylabel('Doğruluk (%)', fontsize=12)
ax.set_title('K Değeri vs Model Doğruluğu', fontsize=14, fontweight='bold')
ax.set_xticks(ks)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim(0, 105)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / '1_k_degeri_dogruluk.png')
plt.close(fig)
print("    1_k_degeri_dogruluk.png")


fig, ax = plt.subplots(figsize=(14, 11))


test_rps = sorted(set(list(y_test) + list(y_pred)), key=lambda x: int(x.replace('RP','')))
cm = confusion_matrix(y_test, y_pred, labels=test_rps)
labels = [f"{rp}\n{rp_coords.loc[rp,'location']}" for rp in test_rps]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, ax=ax,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})

ax.set_xlabel('Tahmin Edilen', fontsize=12)
ax.set_ylabel('Gerçek', fontsize=12)
ax.set_title(f'Confusion Matrix (K={best_k}, Test Doğruluğu: %{test_accuracy*100:.1f})', fontsize=14, fontweight='bold')
ax.tick_params(axis='both', labelsize=7)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / '2_confusion_matrix.png')
plt.close(fig)
print("    2_confusion_matrix.png")


fig, ax = plt.subplots(figsize=(14, 9))

for i in range(len(y)):
    gx = float(rp_coords.loc[y[i], 'pixel_x'])
    gy = float(rp_coords.loc[y[i], 'pixel_y'])
    
    if y[i] == y_self_pred[i]:
        ax.scatter(gx, gy, c='#4CAF50', s=200, edgecolors='darkgreen', 
                   linewidth=2, zorder=5, marker='o')
        ax.annotate(f"{y[i]}\n{rp_coords.loc[y[i],'location']}\n", 
                    (gx, gy), textcoords="offset points", xytext=(0, 18),
                    ha='center', fontsize=6, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#c8e6c9', alpha=0.9))
    else:
        ax.scatter(gx, gy, c='#F44336', s=200, edgecolors='darkred', 
                   linewidth=2, zorder=5, marker='X')
        tx = float(rp_coords.loc[y_self_pred[i], 'pixel_x'])
        ty = float(rp_coords.loc[y_self_pred[i], 'pixel_y'])
        ax.annotate('', xy=(tx, ty), xytext=(gx, gy),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.annotate(f"{y[i]}\n{y_self_pred[i]}", 
                    (gx, gy), textcoords="offset points", xytext=(0, 18),
                    ha='center', fontsize=6, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcdd2', alpha=0.9))

ax.set_xlim(0, 800)
ax.set_ylim(520, 0)
ax.set_title(f'Zemin Kat - KNN Tahmin Haritası (K={best_k})', fontsize=14, fontweight='bold')
ax.set_xlabel('X (piksel)')
ax.set_ylabel('Y (piksel)')
ax.grid(alpha=0.2)
ax.set_facecolor('#f5f5f5')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=12, label='Doru Tahmin'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#F44336', markersize=12, label='Yanl Tahmin'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / '3_tahmin_haritasi.png')
plt.close(fig)
print("    3_tahmin_haritasi.png")


fig, ax = plt.subplots(figsize=(12, 6))
ap_variance = fingerprint.var().sort_values(ascending=True)
colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(ap_variance)))
bars = ax.barh(range(len(ap_variance)), ap_variance.values, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(ap_variance)))
ax.set_yticklabels(ap_variance.index, fontsize=8)
ax.set_xlabel('RSSI Varyansı (yüksek = daha ayırt edici)', fontsize=11)
ax.set_title('AP Önem Sıralaması (RSSI Varyansna Göre)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, ap_variance.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
            f'{val:.1f}', ha='left', va='center', fontsize=8)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / '4_ap_onem_siralamasi.png')
plt.close(fig)
print("    4_ap_onem_siralamasi.png")


fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(fingerprint.round(1), annot=True, fmt='.0f', cmap='RdYlGn',
            ax=ax, linewidths=0.5, vmin=-100, vmax=-40,
            cbar_kws={'label': 'RSSI (dBm)', 'shrink': 0.8},
            annot_kws={'size': 7})
ax.set_title('WiFi Parmak İzi Matrisi (RP x AP)', fontsize=14, fontweight='bold')
ax.set_xlabel('AP (BSSID)')
ax.set_ylabel('Referans Noktası (RP)')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / '5_parmak_izi_haritasi.png')
plt.close(fig)
print("    5_parmak_izi_haritasi.png")


print(f"\n{'='*60}")
print(f"  6. MODEL KAYDETME")
print(f"{'='*60}")

model_data = {
    'model': final_knn,
    'fingerprint_matrix': fingerprint,
    'rp_coords': rp_coords,
    'ap_list': list(fingerprint.columns),
    'rp_list': list(fingerprint.index),
    'best_k': best_k,
    'test_accuracy': test_accuracy,
    'self_accuracy': self_acc,
}

model_path = OUTPUT_DIR / 'knn_ips_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"    Model kaydedildi: {model_path}")
print(f"    Boyut: {model_path.stat().st_size / 1024:.1f} KB")


print(f"\n{'='*60}")
print(f"  7. TAHMNİN DEMO")
print(f"{'='*60}")


print(f"\n   Simlasyon: RP3 (DZ02) noktasndaki sinyal ile tahmin")
test_idx_demo = list(y).index('RP3')
test_signal = X[test_idx_demo]

tahmin = final_knn.predict([test_signal])[0]
proba = final_knn.predict_proba([test_signal])[0]
top3_idx = np.argsort(proba)[::-1][:3]
classes = final_knn.classes_

print(f"   Tahmin: {tahmin} ({rp_coords.loc[tahmin, 'location']})")
print(f"\n   En olas 3 konum:")
for i, idx in enumerate(top3_idx):
    rp_name = classes[idx]
    prob = proba[idx] * 100
    loc_name = rp_coords.loc[rp_name, 'location']
    px = rp_coords.loc[rp_name, 'pixel_x']
    py = rp_coords.loc[rp_name, 'pixel_y']
    print(f"   {i+1}. {rp_name} ({loc_name})  %{prob:.1f}  [pixel: ({px}, {py})]")


print(f"\n   Grltl sinyal ile test (5 dBm):")
for trial in range(3):
    noisy = test_signal + np.random.normal(0, 5, size=test_signal.shape)
    noisy = np.clip(noisy, -100, -20)
    t = final_knn.predict([noisy])[0]
    print(f"   Deneme {trial+1}: {t} ({rp_coords.loc[t, 'location']})", 
          "" if t == 'RP3' else "")


print(f"\n{'='*60}")
print(f"   KNN MODEL TAMAMLANDI!")
print(f"{'='*60}")
print(f"""
    Sonular:
    Optimal K           : {best_k}
    Test doruluu      : %{test_accuracy*100:.1f}
    Self-test doruluu : %{self_acc*100:.1f}
    AP says           : {X.shape[1]}
    RP says           : {X.shape[0]}
    Parmak izi boyutu   : {X.shape}

    ktlar: {OUTPUT_DIR}
    knn_ips_model.pkl
    1_k_degeri_dogruluk.png
    2_confusion_matrix.png
    3_tahmin_haritasi.png
    4_ap_onem_siralamasi.png
    5_parmak_izi_haritasi.png
""")
