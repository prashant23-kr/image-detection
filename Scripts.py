import cv2, time, json, os
import numpy as np
import pyttsx3
from ultralytics import YOLO
import easyocr

try:
    from pyzbar import pyzbar
    HAS_BARCODE = True
except Exception:
    HAS_BARCODE = False

# ---------- Text-to-speech ----------
tts = pyttsx3.init()
tts.setProperty('rate', 175)

def speak(text):
    print("[SAY]:", text)
    try:
        tts.say(text)
        tts.runAndWait()
    except Exception:
        pass

# ---------- Load models ----------
obj_model = YOLO("yolov8n.pt")  # downloads on first use
ocr_reader = easyocr.Reader(['en'], gpu=False)

# ---------- Load local medicine DB ----------
# Example file path; create meds.json like:
# [{"brand":"Dolo 650","generic":"Paracetamol","strength":"650 mg","uses":"Fever, pain","warnings":"Liver disease caution"}]
MED_DB_PATH = "meds.json"
if os.path.exists(MED_DB_PATH):
    with open(MED_DB_PATH, "r", encoding="utf-8") as f:
        MED_DB = json.load(f)
else:
    MED_DB = []

def lookup_medicine_by_name(name_text):
    name_low = name_text.lower()
    best = None
    for r in MED_DB:
        hay = " ".join([r.get("brand",""), r.get("generic",""), r.get("strength","")]).lower()
        if any(tok in hay for tok in name_low.split()):
            best = r
            break
    return best

def lookup_medicine_by_barcode(ean):
    # If you later add a "barcode" field in DB, resolve here.
    for r in MED_DB:
        if r.get("barcode") == ean:
            return r
    return None

# ---------- Modes ----------
MODE_OBJECT = "object"
MODE_MEDICINE = "medicine"

def run():
    cap = cv2.VideoCapture(0)  # use 0 or your device index
    if not cap.isOpened():
        print("Camera not found")
        return

    mode = MODE_OBJECT
    last_spoken = ""
    last_time = 0

    speak("App started. Object mode.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            speak("Goodbye")
            break
        elif key == ord('o'):
            mode = MODE_OBJECT
            speak("Object mode")
            last_spoken = ""
        elif key == ord('m'):
            mode = MODE_MEDICINE
            speak("Medicine mode")
            last_spoken = ""

        display = frame.copy()

        if mode == MODE_OBJECT:
            results = obj_model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
            if len(results.boxes) > 0:
                # Pick highest confidence detection
                confs = results.boxes.conf.cpu().numpy()
                cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                best_i = int(np.argmax(confs))
                label = results.names[cls_ids[best_i]]
                conf = confs[best_i]
                # Draw box and label for UX
                x1,y1,x2,y2 = results.boxes.xyxy[best_i].cpu().numpy().astype(int)
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(display,f"{label} {conf:.2f}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

                # Speak at most once every 2s if changed
                now = time.time()
                if (label != last_spoken or now - last_time > 2.0):
                    speak(label)
                    last_spoken = label
                    last_time = now

        elif mode == MODE_MEDICINE:
            # Try barcode first (fast + accurate)
            found_info = None
            if HAS_BARCODE:
                barcodes = pyzbar.decode(frame)
                for b in barcodes:
                    ean = b.data.decode('utf-8', errors='ignore')
                    cv2.rectangle(display,(b.rect.left,b.rect.top),(b.rect.left+b.rect.width,b.rect.top+b.rect.height),(255,0,0),2)
                    cv2.putText(display, ean, (b.rect.left, b.rect.top-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    found_info = lookup_medicine_by_barcode(ean)
                    if found_info:
                        break

            # If no barcode info, do OCR for name/strength
            if not found_info:
                # Crop center region to reduce noise; you can improve with a detector later
                h, w, _ = frame.shape
                ch, cw = int(h*0.6), int(w*0.9)
                y1, x1 = (h-ch)//2, (w-cw)//2
                crop = frame[y1:y1+ch, x1:x1+cw]
                cv2.rectangle(display,(x1,y1),(x1+cw,y1+ch),(0,200,200),2)

                ocr = ocr_reader.readtext(crop, detail=0, paragraph=True)
                text = " ".join(ocr)
                if text.strip():
                    cv2.putText(display, text[:60]+"...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
                    # Heuristic: find tokens like 250mg, 500 mg etc
                    tokens = text.split()
                    strength = next((t for t in tokens if "mg" in t.lower() or "mcg" in t.lower()), "")
                    # Lookup with fuzzy contain
                    found_info = lookup_medicine_by_name(text)

                    # Speak summary
                    if found_info:
                        summary = f"{found_info.get('brand', found_info.get('generic','Medicine'))} {found_info.get('strength','')}. Uses: {found_info.get('uses','')}. Warnings: {found_info.get('warnings','')}"
                        if summary != last_spoken:
                            speak(summary)
                            last_spoken = summary
                    else:
                        msg = f"Detected text {text[:40]}... {strength}".strip()
                        if msg != last_spoken:
                            speak(msg)
                            last_spoken = msg

        cv2.putText(display, f"Mode: {mode} (o=object, m=medicine, q=quit)", (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(display, f"Mode: {mode} (o=object, m=medicine, q=quit)", (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Assistive Vision MVP", display)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
