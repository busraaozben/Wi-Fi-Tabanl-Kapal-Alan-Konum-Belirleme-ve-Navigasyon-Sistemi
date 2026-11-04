#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEU IPS - Otomatik Canli Konum Testi
"""

import pickle
import numpy as np
import time
import subprocess
import re
from pathlib import Path

MODEL_PATH = Path(r"D:\2026-2027_Guz_Donemi\bilgisayar_muhendisligi_uygulamalari\06 ekim\knn_model_sonuclari\knn_ips_model.pkl")

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model     = model_data['model']
ap_list   = model_data['ap_list']
rp_coords = model_data['rp_coords']
best_k    = model_data['best_k']

print("=" * 55)
print("  NEU IPS - Otomatik Canli Konum Testi")
print("=" * 55)
print(f"  Model: KNN (K={best_k})")
print(f"  Bilinen AP: {len(ap_list)} adet")
print()


def tara_pywifi():
    try:
        import pywifi
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.scan()
        time.sleep(2)
        results = iface.scan_results()

        olcum = {}
        gorulmus = set()
        for ap in results:
            bssid = ap.bssid.strip().rstrip(':').upper()
            rssi  = int(ap.signal)
            ssid  = ap.ssid
            if bssid in gorulmus:
                continue
            gorulmus.add(bssid)
            if 'NEU' in ssid.upper() or bssid in [a.upper() for a in ap_list]:
                olcum[bssid] = rssi
                print(f"  [pywifi] {bssid}  {rssi:>4} dBm  SSID={ssid}")

        return olcum
    except Exception as e:
        print(f"  [pywifi hata]: {e}")
        return None


def tara_netsh():
    try:
        result = subprocess.run(
            ['netsh', 'wlan', 'show', 'networks', 'mode=bssid'],
            capture_output=True, text=True, encoding='cp1254', errors='ignore'
        )
        cikti = result.stdout

        olcum = {}
        satirlar = cikti.splitlines()
        guncel_bssid = None

        for satir in satirlar:
            satir = satir.strip()

            m = re.search(r'BSSID\s+\d+\s*:\s*([0-9a-fA-F:]{17})', satir)
            if m:
                guncel_bssid = m.group(1).upper()
                continue

            if guncel_bssid and 'Signal' in satir:
                m2 = re.search(r'(\d+)%', satir)
                if m2:
                    yuzde = int(m2.group(1))
                    # Cevrim formulu: dBm = (% / 2) - 100
                    dbm = (yuzde / 2) - 100
                    olcum[guncel_bssid] = int(dbm)
                    print(f"  [netsh]  {guncel_bssid}  {int(dbm):>4} dBm  (sinyal %{yuzde})")
                    guncel_bssid = None

        return olcum if olcum else None
    except Exception as e:
        print(f"  [netsh hata]: {e}")
        return None


def konum_tahmin_et(olcum):
    if not olcum:
        return

    vektor = []
    eslesen = 0
    for ap in ap_list:
        deger = -100
        for bssid, rssi in olcum.items():
            if bssid.upper() == ap.upper():
                deger = rssi
                eslesen += 1
                break
        vektor.append(deger)

    vektor = np.array(vektor).reshape(1, -1)

    print()
    print(f"  Taranan AP    : {len(olcum)}")
    print(f"  Modelle eslesen: {eslesen} / {len(ap_list)}")

    if eslesen == 0:
        print()
        print("  [!] Hic bilinen AP bulunamadi.")
        print("      Kampuste degilsiniz ya da sinyal cok zayif.")
        return

    # Tahmin
    tahmin_rp = model.predict(vektor)[0]
    proba     = model.predict_proba(vektor)[0]
    classes   = model.classes_
    top3_idx  = np.argsort(proba)[::-1][:3]

    konum   = rp_coords.loc[tahmin_rp, 'location']
    pixel_x = rp_coords.loc[tahmin_rp, 'pixel_x']
    pixel_y = rp_coords.loc[tahmin_rp, 'pixel_y']
    guvven  = proba[list(classes).index(tahmin_rp)] * 100

    print()
    print("=" * 55)
    print(f"  >> TAHMIN: {tahmin_rp} - {konum}")
    print(f"     Guven : %{guvven:.1f}")
    print(f"     Harita: ({pixel_x}, {pixel_y})")
    print("=" * 55)
    print()
    print("  En olasilik 3 konum:")
    print(f"  {'Sira':<5} {'RP':<6} {'Konum':<22} {'Olasilik':<10} {'Bar'}")
    print(f"  {'-'*5} {'-'*6} {'-'*22} {'-'*10} {'-'*20}")
    for sira, idx in enumerate(top3_idx, 1):
        rp_name  = classes[idx]
        prob     = proba[idx] * 100
        loc_name = rp_coords.loc[rp_name, 'location']
        bar      = '|' * int(prob / 5)
        print(f"  {sira:<5} {rp_name:<6} {loc_name:<22} %{prob:5.1f}     {bar}")


SUREKLI_MOD = False

dongu = 0
while True:
    dongu += 1
    print(f"\n--- Tarama #{dongu} ---")
    print("  WiFi taraniyor (2-3 saniye)...")

    olcum = tara_pywifi()
    if not olcum:
        print("  pywifi calismadi, netsh deneniyor...")
        olcum = tara_netsh()

    konum_tahmin_et(olcum)

    if not SUREKLI_MOD:
        break

    print()
    print("  [5 saniye sonra yeniden taranacak... Cikmak icin Ctrl+C]")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n  Cikiliyor...")
        break
