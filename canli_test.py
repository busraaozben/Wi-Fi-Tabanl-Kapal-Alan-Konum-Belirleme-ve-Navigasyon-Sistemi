#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEU IPS - Canli Konum Testi
"""

import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path(r"D:\2026-2027_Guz_Donemi\bilgisayar_muhendisligi_uygulamalari\06 ekim\knn_model_sonuclari\knn_ips_model.pkl")


with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model       = model_data['model']
ap_list     = model_data['ap_list']
rp_coords   = model_data['rp_coords']
best_k      = model_data['best_k']
train_acc   = model_data['test_accuracy']

print("=" * 55)
print("  NEU IPS - Canli Konum Tahmin Testi")
print("=" * 55)
print(f"  Model: KNN (K={best_k})")
print(f"  Egitim dogrulugu: %{train_acc*100:.1f}")
print(f"  Bilinen AP sayisi: {len(ap_list)}")
print()


canli_olcum = {
    # ------- BURAYA TELEFONDAN ALDIGINI BSSID:RSSI DEGERLERINI GIR -------
    '6C:14:6E:AC:44:42': -80,
    '6C:14:6E:AC:72:22': -88,
    '6C:14:6E:52:35:02': -93,
    '6C:14:6E:52:5B:02': -41,
    '6C:14:6E:52:3A:02': -90,
    '6C:14:6E:51:F9:42': -77,
    '6C:14:6E:52:24:22': -69,
    '9C:B2:E8:97:45:B2': -92



    # '6C:14:6E:52:39:E2': -68,
    # '9C:B2:E8:97:47:A2': -70,
    # '6C:14:6E:AC:72:12': -73
}


olcum_vektor = []
for ap in ap_list:
    deger = -100
    for bssid, rssi in canli_olcum.items():
        if bssid.upper() == ap.upper():
            deger = rssi
            break
    olcum_vektor.append(deger)

olcum_vektor = np.array(olcum_vektor).reshape(1, -1)


eslesen = sum(1 for ap in ap_list
              for bssid in canli_olcum
              if bssid.upper() == ap.upper())

print(f"  Girilen AP sayisi  : {len(canli_olcum)}")
print(f"  Modelle eslesen    : {eslesen} / {len(ap_list)}")

if eslesen == 0:
    print()
    print("  [UYARI] Hic AP eslesmiyor!")
    print("  Lutfen canli_olcum sozlugune degerler girin.")
    print()
else:
    tahmin_rp = model.predict(olcum_vektor)[0]
    proba     = model.predict_proba(olcum_vektor)[0]
    classes   = model.classes_

    top3_idx  = np.argsort(proba)[::-1][:3]

    konum   = rp_coords.loc[tahmin_rp, 'location']
    pixel_x = rp_coords.loc[tahmin_rp, 'pixel_x']
    pixel_y = rp_coords.loc[tahmin_rp, 'pixel_y']

    print()
    print("=" * 55)
    print(f"  TAHMIN EDILEN KONUM: {tahmin_rp} - {konum}")
    print(f"  Harita koordinati : ({pixel_x}, {pixel_y})")
    print("=" * 55)
    print()
    print("  En olasilik 3 konum:")
    print(f"  {'Sira':<5} {'RP':<6} {'Konum':<25} {'Olasilik'}")
    print(f"  {'-'*5} {'-'*6} {'-'*25} {'-'*10}")
    for sira, idx in enumerate(top3_idx, 1):
        rp_name  = classes[idx]
        prob     = proba[idx] * 100
        loc_name = rp_coords.loc[rp_name, 'location']
        bar      = '#' * int(prob / 5)
        print(f"  {sira:<5} {rp_name:<6} {loc_name:<25} %{prob:5.1f}  {bar}")


print()
print("=" * 55)
print("  Modelin bildigi AP'ler (bunlari telefonda gir):")
print("=" * 55)
for i, ap in enumerate(ap_list, 1):
    print(f"  {i:>2}. {ap}")
print()
print("  WiFi Data / WiFi Analyzer uygulamasinda")
print("  NEU_OGRENCI satirlarini yukardaki listesiyle eslestirir")
print("  'canli_olcum' sozlugune girerek testi calistir.")
