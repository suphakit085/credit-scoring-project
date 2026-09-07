# ขั้นตอนการทำงานของ Credit Scoring Project

เอกสารนี้อธิบายการทำงานของโปรเจกต์ตั้งแต่ข้อมูลดิบ การเตรียมข้อมูล
การสร้างและประเมินโมเดล ไปจนถึงการทำนายผ่าน Streamlit

## 1. วัตถุประสงค์

ระบบใช้ชุดข้อมูล Home Credit Default Risk เพื่อประมาณ
**ความน่าจะเป็นที่ผู้ขอสินเชื่อจะผิดนัดชำระหนี้**

- TARGET = 0 หมายถึง ไม่ผิดนัดชำระ
- TARGET = 1 หมายถึง ผิดนัดชำระ

ผลลัพธ์ประกอบด้วย probability ในการผิดนัด ระดับความเสี่ยง
และดัชนีความเสี่ยงช่วง 300–850 ดัชนีดังกล่าวเป็นการนำเสนอแบบ odds-based
ภายในโปรเจกต์ ไม่ใช่คะแนนเครดิตบูโร

ระบบนี้เป็นตัวอย่างด้าน Machine Learning ไม่ควรใช้เป็นคำตัดสินอนุมัติสินเชื่อ
โดยไม่มี human review และการตรวจสอบด้านกฎหมาย ความเป็นธรรม และคุณภาพข้อมูล

## 2. ภาพรวม Data Flow

~~~text
ข้อมูลดิบใน data/raw
        |
        v
ทำความสะอาดและรวมตาราง
src/data_preprocessing.py
        |
        v
data/processed/train_cleaned.csv
data/processed/test_cleaned.csv
        |
        v
แบ่ง Train / Calibration / Validation
        |
        v
สร้างฟีเจอร์ -> Impute -> Encode -> Scale
        |
        v
ตัดฟีเจอร์ค่าคงที่ -> เลือก 200 ฟีเจอร์
        |
        v
ฝึก LightGBM -> Calibrate Probability -> เลือก Threshold
        |
        v
models/credit_scoring_bundle.joblib
        |
        v
src/inference.py -> app/streamlit_app.py
~~~

## 3. ข้อมูลดิบ

ไฟล์หลักอยู่ในโฟลเดอร์ data/raw:

| ไฟล์ | หน้าที่ |
|---|---|
| application_train.csv | ข้อมูลผู้สมัครสินเชื่อที่มี TARGET |
| application_test.csv | ข้อมูลผู้สมัครสำหรับทำนาย |
| bureau.csv | ประวัติสินเชื่อจากเครดิตบูโร |
| previous_application.csv | ประวัติการสมัครสินเชื่อก่อนหน้า |
| POS_CASH_balance.csv | ประวัติสินเชื่อ POS/Cash |
| credit_card_balance.csv | ประวัติบัตรเครดิต |
| installments_payments.csv | ประวัติการผ่อนชำระ |

ข้อมูลดิบมีขนาดใหญ่และไม่ถูกบันทึกลง Git ตามกฎใน .gitignore

## 4. การเตรียมข้อมูล

ไฟล์ที่รับผิดชอบคือ src/data_preprocessing.py

### 4.1 ทำความสะอาด Application Data

ค่า 365243 ใน DAYS_EMPLOYED เป็น sentinel ที่ใช้แทนข้อมูลผิดปกติ ระบบจะ:

1. สร้าง DAYS_EMPLOYED_ANOM เพื่อบันทึกว่าพบ sentinel
2. แทนค่า 365243 ด้วย missing value
3. ยังไม่เติม missing value ในขั้นตอนนี้

การไม่เติม missing value ก่อนแบ่งข้อมูลเป็นสิ่งสำคัญ เพราะ median หรือ mode
ต้องคำนวณจาก training split เท่านั้นเพื่อป้องกัน data leakage

### 4.2 Aggregate ตารางประวัติ

ตารางประวัติมีหลายแถวต่อผู้สมัคร ระบบ group ตาม SK_ID_CURR
แล้วคำนวณสถิติของคอลัมน์ตัวเลข:

- count
- mean
- max
- min
- sum
- standard deviation

ตัวอย่างชื่อฟีเจอร์:

- BUREAU_DAYS_CREDIT_mean
- PREV_AMT_CREDIT_max
- CC_AMT_BALANCE_sum
- INSTALL_AMT_PAYMENT_std

### 4.3 รวมข้อมูล

Aggregate ทุกตารางจะถูก left join กลับเข้า application data ด้วย SK_ID_CURR
พร้อมตรวจสอบว่าผลลัพธ์ยังมีหนึ่งแถวต่อผู้สมัคร

ผลลัพธ์ปัจจุบัน:

- Training: 307,511 แถว 495 คอลัมน์
- Test: 48,744 แถว 494 คอลัมน์

ระบบเขียนผ่าน temporary file ก่อนแทนที่ไฟล์ปลายทาง
เพื่อลดความเสี่ยงจากไฟล์เสียหายหากกระบวนการหยุดกลางทาง

คำสั่งสร้างข้อมูล:

~~~powershell
..venvScriptspython.exe src/data_preprocessing.py
~~~

หาก aggregate files มีอยู่แล้ว:

~~~powershell
..venvScriptspython.exe src/data_preprocessing.py --reuse-aggregates
~~~

## 5. การตรวจสอบคุณภาพข้อมูล

ใช้ src/check_data_quality.py เพื่อตรวจ:

- คอลัมน์บังคับ SK_ID_CURR, TARGET และ DAYS_EMPLOYED
- SK_ID_CURR ต้องไม่ซ้ำ
- TARGET ต้องมีเฉพาะ 0 และ 1
- sentinel 365243 ต้องถูกนำออกแล้ว
- ข้อมูลตัวเลขต้องไม่มี positive/negative infinity
- สัดส่วน TARGET
- จำนวนคอลัมน์ที่มี missing valueมากกว่า 50%

~~~powershell
..venvScriptspython.exe src/check_data_quality.py
~~~

ตรวจเพียงบางแถวระหว่างพัฒนา:

~~~powershell
..venvScriptspython.exe src/check_data_quality.py --sample-rows 20000
~~~

## 6. การแบ่งข้อมูลโดยป้องกัน Data Leakage

ขั้นตอนฝึกอยู่ใน src/model_training.py ข้อมูลถูกแบ่งก่อน preprocessing
ทุกขั้นตอนที่ต้องเรียนรู้ค่าจากข้อมูล:

| ชุดข้อมูล | สัดส่วน | หน้าที่ |
|---|---:|---|
| Training | 70% | Fit imputer, encoder, scaler, selector และ LightGBM |
| Calibration | 15% | Calibrate probability และเลือก decision threshold |
| Validation | 15% | ประเมินผลขั้นสุดท้ายเท่านั้น |

ใช้ random_state = 42 และ stratified split เพื่อรักษาสัดส่วน TARGET

pipeline เดิมทำ SMOTE ก่อนแบ่ง validation ทำให้ข้อมูลสังเคราะห์ที่สัมพันธ์กัน
อาจกระจายอยู่ทั้ง train และ validation ส่งผลให้คะแนนสูงเกินจริง
pipeline ปัจจุบันจึงไม่ใช้ SMOTE และประเมินด้วยสัดส่วน class ตามข้อมูลจริง

## 7. การสร้างฟีเจอร์

สูตรอยู่ใน src/feature_generation.py และถูกเรียกใช้ร่วมกันทั้งตอนฝึกและทำนาย
จึงลดปัญหา training-serving skew

### 7.1 Financial Ratios

- CREDIT_TO_ANNUITY_RATIO
- CREDIT_TO_GOODS_RATIO
- ANNUITY_TO_INCOME_RATIO
- CREDIT_TO_INCOME_RATIO
- INCOME_PER_PERSON

หากตัวหารเป็นศูนย์ ระบบจะสร้าง missing value ให้ imputer จัดการ
และไม่ปล่อยค่า infinity เข้าโมเดล

### 7.2 Time Features

- AGE_YEARS
- EMPLOYMENT_YEARS
- REGISTRATION_YEARS
- ID_PUBLISH_YEARS
- EMPLOYMENT_TO_AGE_RATIO

### 7.3 External Credit Features

จาก EXT_SOURCE_1, EXT_SOURCE_2 และ EXT_SOURCE_3:

- EXT_SOURCE_MEAN
- EXT_SOURCE_STD
- EXT_SOURCE_MIN
- EXT_SOURCE_MAX

### 7.4 ฟีเจอร์เพิ่มเติม

- จำนวนเอกสารทั้งหมด
- ค่าเฉลี่ยเครดิตจาก Bureau
- จำนวนรายการ Bureau รวม

## 8. Preprocessing และ Feature Selection

Preprocessor เรียนรู้จาก training split เท่านั้น

### คอลัมน์ตัวเลข

1. เติม missing value ด้วย median
2. เพิ่ม missing indicator
3. ปรับสเกลด้วย RobustScaler

### คอลัมน์ข้อความ

1. เติม missing value ด้วยค่าที่พบบ่อยที่สุด
2. แปลงด้วย OneHotEncoder
3. Category ใหม่ที่ไม่เคยพบตอนฝึกจะไม่ทำให้ระบบ error

### การเลือกฟีเจอร์

1. ใช้ VarianceThreshold ตัดคอลัมน์ค่าคงที่
2. บังคับเก็บฟีเจอร์ทั้งหมดที่ควบคุมโดยช่องกรอกใน Streamlit
3. เติมฟีเจอร์ที่เหลือตามคะแนน ANOVA F-test
4. จำกัดจำนวนสุดท้ายไว้ที่ 200 ฟีเจอร์

จึงมั่นใจได้ว่ารายได้ จำนวนบุตร อสังหาริมทรัพย์ ยอดผ่อน
และข้อมูลอื่นที่ผู้ใช้กรอกถูกส่งเข้าโมเดลจริง

## 9. การฝึก LightGBM

โมเดลหลักคือ LightGBMClassifier ซึ่งเหมาะกับข้อมูลตาราง
และรองรับความสัมพันธ์แบบไม่เป็นเส้นตรง

ค่าหลัก:

- estimators: 400
- learning rate: 0.03
- leaves: 31
- minimum child samples: 40
- column sampling: 0.85
- L1/L2 regularization
- random state: 42

ฝึกด้วยข้อมูลเต็ม:

~~~powershell
..venvScriptspython.exe src/model_training.py
~~~

Smoke test ด้วยข้อมูลบางส่วน:

~~~powershell
..venvScriptspython.exe src/model_training.py --sample-rows 20000 --output models/dev_bundle.joblib
~~~

ห้ามนำ metric จาก smoke test ไปใช้อ้างอิงเป็นผลของโมเดลเต็ม

## 10. Probability Calibration และ Threshold

Probability ดิบจาก classifier อาจไม่ตรงกับอัตราผิดนัดจริง
ระบบจึงใช้ IsotonicRegression กับ calibration split

หลัง calibration ระบบเลือก threshold ที่ให้ F1 score สูงที่สุดบน calibration split
แล้วนำ threshold ไปประเมินเพียงครั้งเดียวบน validation split

ผลการฝึกล่าสุด:

| Metric | ค่า |
|---|---:|
| ROC-AUC | 0.7756 |
| Brier score | 0.0667 |
| Decision threshold | 0.1724 |
| Accuracy | 0.8460 |
| Precision | 0.2516 |
| Recall | 0.4597 |
| F1 | 0.3252 |

ROC-AUC วัดความสามารถในการจัดลำดับความเสี่ยง ส่วน Brier score
วัดความคลาดเคลื่อนของ probability โดยค่ายิ่งต่ำยิ่งดี

## 11. Model Bundle

ผลการฝึกอยู่ใน models/credit_scoring_bundle.joblib ซึ่งประกอบด้วย:

- Bundle version และเวลาที่ฝึก
- รายชื่อ input columns
- Default values จาก training split
- Imputer, encoder และ scaler
- Variance filter
- Selected feature indices และ names
- LightGBM model
- Isotonic calibrator
- Decision และ risk thresholds
- Validation metrics
- Training metadata

ไฟล์ประกอบ:

- models/model_metrics.json
- models/feature_importance.csv

การรวมองค์ประกอบใน bundle เดียวช่วยป้องกันการใช้ scaler,
feature order หรือ threshold ผิดรุ่น

## 12. ขั้นตอนทำนาย

ฟังก์ชัน inference อยู่ใน src/inference.py เมื่อรับข้อมูลผู้สมัคร ระบบจะ:

1. โหลด model bundle
2. สร้างหนึ่งแถวตาม input schema ตอนฝึก
3. เติมฟีเจอร์ที่ไม่มีด้วย default จาก training split
4. เขียนทับ default ด้วยค่าจากผู้ใช้หรือ backend
5. เรียกสูตร feature generation ชุดเดียวกับตอนฝึก
6. Transform ด้วย preprocessor
7. ใช้ variance filter และเลือก 200 ฟีเจอร์
8. ทำนาย probability ดิบด้วย LightGBM
9. Calibrate probability
10. แปลง probability เป็นระดับความเสี่ยงและดัชนี 300–850

ฟีเจอร์ประวัติเครดิตที่หน้าเว็บไม่ได้รับจากผู้ใช้จะใช้ training default
ใน production ต้องเปลี่ยนเป็นข้อมูลจริงจากแหล่งที่ได้รับอนุญาต

## 13. การทำงานของ Streamlit

หน้าเว็บอยู่ใน app/streamlit_app.py แบบฟอร์มรับ:

- ข้อมูลส่วนตัวและครอบครัว
- รายได้ วงเงิน ยอดผ่อน และราคาสินค้า
- การทำงาน อาชีพ และทรัพย์สิน
- External credit scores

เมื่อกด **วิเคราะห์ความเสี่ยง**:

1. แปลงค่าจากแบบฟอร์มเป็นชื่อคอลัมน์ Home Credit
2. เรียก predict_default_probability
3. แสดง calibrated probability
4. แสดงระดับความเสี่ยงตาม threshold ใน bundle
5. แสดงดัชนีแบบ odds-based

~~~powershell
..venvScriptspython.exe -m streamlit run app/streamlit_app.py
~~~

เปิดที่ http://localhost:8501

## 14. Tests

Tests อยู่ในโฟลเดอร์ tests และครอบคลุม:

- สูตร financial, time และ external-source features
- การป้องกัน division by zero และ infinity
- การแทน sentinel ของ DAYS_EMPLOYED
- การ aggregate ตารางประวัติ
- การสร้าง input row จาก defaults
- ความเป็น monotonic และขอบเขตของดัชนี 300–850
- การจัดระดับความเสี่ยง

~~~powershell
..venvScriptspython.exe -m pytest -q
~~~

ผลล่าสุด: **9 passed**

## 15. ลำดับคำสั่งตั้งแต่เริ่มต้น

~~~powershell
# 1. สร้าง environment
python -m venv .venv
..venvScriptsActivate.ps1

# 2. ติดตั้ง dependencies
python -m pip install -r requirements.txt

# 3. เตรียมข้อมูล
python src/data_preprocessing.py --reuse-aggregates

# 4. ตรวจคุณภาพ
python src/check_data_quality.py

# 5. ฝึกและประเมินโมเดล
python src/model_training.py

# 6. รัน tests
python -m pytest -q

# 7. เปิดเว็บ
python -m streamlit run app/streamlit_app.py
~~~

## 16. สิ่งที่ต้องเพิ่มก่อน Production

1. Out-of-time validation และ external validation
2. Fairness analysis แยกตามกลุ่มประชากร
3. Monitoring ของ data drift และ model drift
4. การจัดการ consent และข้อมูลส่วนบุคคล
5. Model registry และการบันทึก version ต่อคำทำนาย
6. Audit log และเหตุผลประกอบการตัดสินใจ
7. เชื่อมต่อ Bureau และ Payment history จริง
8. Policy สำหรับ human review และ override
9. Security review สำหรับการโหลดไฟล์ Joblib
10. ประเมินต้นทุน False Positive/False Negative เพื่อเลือก threshold ทางธุรกิจ

Notebooks ในโฟลเดอร์ notebooks เป็นประวัติการทดลองเดิม ส่วน source of truth
สำหรับ pipeline ปัจจุบันคือไฟล์ใน src และ model bundle รุ่นล่าสุด
