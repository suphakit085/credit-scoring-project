# Credit Scoring Project

ระบบประเมินความน่าจะเป็นในการผิดนัดชำระหนี้จากชุดข้อมูล Home Credit Default
Risk พร้อม data pipeline, leakage-safe model training, probability calibration
และหน้าเว็บ Streamlit

คำอธิบายขั้นตอนทั้งหมดแบบละเอียด: [PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md)

> ระบบนี้เป็นตัวอย่างด้าน Machine Learning เท่านั้น ไม่ควรใช้อนุมัติหรือปฏิเสธ
> สินเชื่อจริงโดยไม่มีการตรวจสอบ fairness, compliance, data governance
> และ human review

## สถาปัตยกรรม

~~~text
data/raw/*.csv
    |
    v
src/data_preprocessing.py
    |  clean sentinel + aggregate 5 historical tables
    v
data/processed/train_cleaned.csv
    |
    v
src/model_training.py
    |  split -> defaults/features/preprocess/select (fit on train only)
    |  -> LightGBM -> isotonic calibration -> untouched validation
    v
models/credit_scoring_bundle.joblib
    |
    v
src/inference.py -> app/streamlit_app.py
~~~

Model bundle เก็บ input schema, training-set defaults, imputer, encoder, scaler,
feature selector, LightGBM, probability calibrator และ decision threshold ไว้
ด้วยกัน จึงไม่มีการสร้าง preprocessing แยกจากโมเดลในหน้าแอป

## ติดตั้ง

ต้องใช้ Python 3.10 ขึ้นไป

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

## การใช้งาน

เตรียมข้อมูลจากไฟล์ Kaggle ใน data/raw:

~~~powershell
python src/data_preprocessing.py --reuse-aggregates
python src/check_data_quality.py
~~~

ฝึกโมเดลใหม่:

~~~powershell
python src/model_training.py
~~~

ระหว่างพัฒนา สามารถ smoke test ด้วยข้อมูลบางส่วนได้:

~~~powershell
python src/model_training.py --sample-rows 20000 --output models/dev_bundle.joblib
~~~

เปิดเว็บ:

~~~powershell
streamlit run app/streamlit_app.py
~~~

ตรวจสอบโมเดลและรัน tests:

~~~powershell
python src/inspect_model.py
python -m pytest -q
~~~

## การประเมินที่ถูกต้อง

- แบ่ง train, calibration และ validation ก่อน imputation, encoding, scaling
  และ feature selection
- ไม่ใช้ SMOTE ใน validation และเก็บสัดส่วน class ตามข้อมูลจริง
- ใช้ calibration split สำหรับ Isotonic Regression และเลือก threshold
- รายงาน ROC-AUC, Brier score, precision, recall และ F1 บน validation
  ที่ไม่ถูกใช้ฝึกหรือ calibrate
- ผลล่าสุดถูกบันทึกใน models/model_metrics.json

Notebooks เดิมเก็บไว้เป็นบันทึกการทดลอง แต่ pipeline ที่รองรับการใช้งานจริงคือ
โค้ดใน src/ โดย notebook results เดิมที่ทำ SMOTE ก่อน split ไม่ควรใช้เป็น
ตัวเลขอ้างอิง

## โครงสร้างสำคัญ

~~~text
app/streamlit_app.py       หน้าเว็บและแบบฟอร์ม
src/data_preprocessing.py  รวมข้อมูลดิบ
src/feature_generation.py  สูตรฟีเจอร์ที่ train/inference ใช้ร่วมกัน
src/model_training.py      ฝึก calibrate ประเมิน และบันทึก bundle
src/inference.py           API ทำนายที่หน้าเว็บเรียกใช้
src/check_data_quality.py  ตรวจ schema, ID, target, sentinel และ infinity
tests/                     unit tests
notebooks/                 ประวัติการสำรวจและทดลองเดิม
~~~

ฟีเจอร์เครดิตบูโรและประวัติการชำระที่หน้าเว็บไม่ได้รับจากผู้ใช้จะใช้ค่ากลาง
ของ training split ใน demo นี้ ระบบ production ต้องเชื่อมต่อแหล่งข้อมูลจริง
ที่ได้รับอนุญาตและบันทึก model/data version สำหรับทุกคำทำนาย
