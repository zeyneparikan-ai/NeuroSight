import cv2
from ultralytics import YOLO
from gtts import gTTS
import playsound
import threading
import os
import time

# 1. Yapay Zeka Modelini ve Kamerayı Başlatıyoruz
model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0)

# Ses dosyalarının çakışmaması için benzersiz isim üreten fonksiyon
def seslendir(metin):
    try:
        tts = gTTS(text=metin, lang='tr')
        dosya_adi = f"uyari_{int(time.time())}.mp3"
        tts.save(dosya_adi)
        playsound.playsound(dosya_adi)
        os.remove(dosya_adi) # İş bitince geçici ses dosyasını siliyoruz
    except Exception as e:
        pass

# Aynı sesin üst üste binmesini engellemek için zaman kontrolü
son_sesme_zamani = 0

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        yukseklik, genislik, _ = frame.shape
        ekran_merkezi = genislik // 2
        toplam_ekran_alani = yukseklik * genislik
        
        # Yapay Zeka Nesne Tespiti
        results = model(frame, stream=True)
        tehlike_mesaji = ""
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                nesne_adi = model.names[cls]
                
                # Günlük hayattaki kritik engeller
                engeller = ["person", "cell phone", "cup", "bottle", "chair"]
                if nesne_adi in engeller:
                    
                    # Konum ve Mesafe Algoritmaları
                    kutu_alani = (x2 - x1) * (y2 - y1)
                    alan_yuzdesi = (kutu_alani / toplam_ekran_alani) * 100
                    kutu_merkezi_x = (x1 + x2) // 2
                    
                    if kutu_merkezi_x < (ekran_merkezi - 120):
                        yon = "solda"
                    elif kutu_merkezi_x > (ekran_merkezi + 120):
                        yon = "sağda"
                    else:
                        yon = "merkezde"
                    
                    # Tehlike Analizi (%18'den büyükse yakındır)
                    if alan_yuzdesi > 18:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(frame, f"TEHLIKE! {yon.upper()} YAKINDA {nesne_adi.upper()}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Türkçe dönüştürme sözlüğü
                        turkce_isimler = {"person": "insan", "cell phone": "telefon", "cup": "bardak", "bottle": "şişe", "chair": "sandalye"}
                        nesne_tr = turkce_isimler.get(nesne_adi, nesne_adi)
                        tehlike_mesaji = f"Dikkat, {yon} yakın mesafede {nesne_tr} var."
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{nesne_adi} ({yon})", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Sesli Uyarıyı Arka Planda Tetikleme (Donma Yapmaz)
        simdiki_zaman = time.time()
        if tehlike_mesaji and (simdiki_zaman - son_sesme_zamani > 10): # 10 saniyede bir konuşur
            threading.Thread(target=seslendir, args=(tehlike_mesaji,), daemon=True).start()
            son_sesme_zamani = simdiki_zaman
            
        cv2.imshow("NeuroSight - Tamamlanmis Yapay Zeka Projesi", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
