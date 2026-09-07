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
    |  scope to servable columns -> 4-way split
    |  -> defaults/features/preprocess/select (fit on train only)
    |  -> LightGBM -> isotonic calibration -> independent threshold split
    |  -> untouched validation
    v
models/credit_scoring_bundle.joblib
    |
    v
src/inference.py -> app/streamlit_app.py
~~~

Model bundle เก็บ input schema, รายชื่อคอลัมน์ที่หน้าเว็บต้องส่ง, training-set
defaults, imputer, encoder, scaler, feature selector, LightGBM, probability
calibrator และ decision threshold ไว้ด้วยกัน จึงไม่มีการสร้าง preprocessing
แยกจากโมเดลในหน้าแอป

โมเดลถูกจำกัดขอบเขตให้ใช้เฉพาะคอลัมน์ที่หน้าเว็บเก็บได้จริง ผ่าน
`--feature-scope` ซึ่งค่าเริ่มต้นคือ `form` การเทรนด้วยฟีเจอร์ที่ตอนใช้งานจริง
ถูกตรึงไว้ที่ค่ามัธยฐาน ทำให้ตัวเลข offline อธิบายระบบคนละตัวกับที่ deploy จริง

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

เลือกขอบเขตฟีเจอร์ได้ด้วย `--feature-scope` ค่าเริ่มต้น `form` คือใช้เฉพาะ
คอลัมน์ที่หน้าเว็บเก็บได้ ส่วน `application` และ `full` ใช้เมื่อมีช่องทาง
ป้อนข้อมูลเพิ่มเติมแล้วเท่านั้น

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
pytest
~~~

## การประเมินที่ถูกต้อง

- แบ่ง train, calibration, threshold และ validation ก่อน imputation, encoding,
  scaling และ feature selection
- ไม่ใช้ SMOTE ใน validation และเก็บสัดส่วน class ตามข้อมูลจริง
- ใช้ calibration split สำหรับ Isotonic Regression เท่านั้น และเลือก threshold
  บน split อีกชุดที่ calibrator ไม่เคยเห็น
- รายงาน ROC-AUC, average precision, Brier score, precision, recall และ F1
  บน validation ที่ไม่ถูกใช้ฝึก calibrate หรือเลือก threshold
- ผลล่าสุดถูกบันทึกใน models/credit_scoring_bundle_metrics.json

ผลล่าสุด scope `form` 22 คอลัมน์ 150 ฟีเจอร์

| Metric | ค่า |
|---|---:|
| ROC-AUC | 0.7665 |
| Average precision | 0.2416 |
| Brier score | 0.0674 |
| Decision threshold | 0.1665 |
| Precision | 0.2498 |
| Recall | 0.4060 |
| F1 | 0.3092 |

ตัวเลขนี้เทียบตรงๆ กับ 0.7756 ที่เคยรายงานไม่ได้ ค่าเดิมมาจากโมเดล 495 คอลัมน์
ที่วัดตอนมีข้อมูลครบทุกช่อง ซึ่งหน้าเว็บไม่เคยมี เมื่อวัดโมเดลตัวเดิมในสภาพ
ที่หน้าเว็บเสิร์ฟจริงจะได้ 0.7612 ต่ำกว่าโมเดลปัจจุบัน

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

ใน scope `form` โมเดลไม่ใช้ฟีเจอร์เครดิตบูโรและประวัติการชำระเลย จึงไม่มี
ฟีเจอร์ใดถูกตรึงที่ค่ากลางของ training split อีก หากภายหลังเปลี่ยนไปใช้ scope
`application` หรือ `full` ฟีเจอร์ที่หน้าเว็บไม่ได้รับจะกลับไปใช้ค่า default
ซึ่งต้องเปลี่ยนเป็นการเชื่อมต่อแหล่งข้อมูลจริงที่ได้รับอนุญาตก่อน

ช่อง EXT_SOURCE ทั้งสามเป็นค่าที่เจ้าหน้าที่สินเชื่อกรอกจากระบบเครดิตภายนอก
ไม่ใช่ค่าที่ผู้สมัครประเมินเอง แต่ละช่องมีตัวเลือกไม่พบในระบบซึ่งส่งค่าว่าง
เข้าแบบจำลอง ตรงกับข้อมูลฝึกที่ EXT_SOURCE_1 ขาดหายถึง 56%

ระบบ production ยังต้องบันทึก model/data version สำหรับทุกคำทำนาย
