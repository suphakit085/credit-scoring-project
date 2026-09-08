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
แบ่ง Train / Calibration / Threshold / Validation
        |
        v
สร้างฟีเจอร์ -> Impute -> Encode -> Scale
        |
        v
ตัดฟีเจอร์ค่าคงที่ -> เลือกฟีเจอร์ตามขอบเขตที่เสิร์ฟได้
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
python src/data_preprocessing.py
~~~

หาก aggregate files มีอยู่แล้ว:

~~~powershell
python src/data_preprocessing.py --reuse-aggregates
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
python src/check_data_quality.py
~~~

ตรวจเพียงบางแถวระหว่างพัฒนา:

~~~powershell
python src/check_data_quality.py --sample-rows 20000
~~~

## 6. การแบ่งข้อมูลโดยป้องกัน Data Leakage

ขั้นตอนฝึกอยู่ใน src/model_training.py ข้อมูลถูกแบ่งก่อน preprocessing
ทุกขั้นตอนที่ต้องเรียนรู้ค่าจากข้อมูล:

| ชุดข้อมูล | สัดส่วน | หน้าที่ |
|---|---:|---|
| Training | 70% | Fit imputer, encoder, scaler, selector และ LightGBM |
| Calibration | 10% | Fit Isotonic Regression อย่างเดียว |
| Threshold | 10% | เลือก decision threshold อย่างเดียว |
| Validation | 10% | ประเมินผลขั้นสุดท้ายเท่านั้น |

เดิมใช้สามชุดโดยให้ calibration split ทำสองหน้าที่พร้อมกัน คือ fit calibrator
และเลือก threshold บนชุดเดียวกัน ทำให้ threshold ถูกเลือกจาก probability
ที่ calibrator เห็นมาแล้ว จุดตัดที่ได้จึงมองโลกในแง่ดีเกินจริง ปัจจุบันแยกเป็น
สี่ชุด แต่ละชุดทำหน้าที่เดียว

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

### ขอบเขตฟีเจอร์

`--feature-scope` กำหนดว่าโมเดลจะเรียนรู้จากคอลัมน์กลุ่มไหน ค่าเริ่มต้นคือ `form`

| scope | คอลัมน์ | ใช้เมื่อไร |
|---|---|---|
| `form` | เฉพาะ 22 คอลัมน์ที่หน้าเว็บเก็บได้ | ค่าเริ่มต้น ตรงกับสิ่งที่เสิร์ฟจริง |
| `application` | ตาราง application ทั้งหมด ไม่รวม aggregate | เมื่อขยายฟอร์มหรือรับไฟล์นำเข้า |
| `full` | ทุกคอลัมน์รวมประวัติ | ต่อข้อมูล Bureau จริงแล้วเท่านั้น |

เหตุผลที่ค่าเริ่มต้นเป็น `form` วัดจากข้อมูลชุดนี้โดยตรง โมเดล 495 คอลัมน์
ได้ ROC-AUC 0.7756 ตอนมีข้อมูลครบ แต่เหลือ 0.7612 เมื่อหน้าเว็บเสิร์ฟจริง
เพราะ 137 จาก 200 ฟีเจอร์ถูกตรึงที่ค่ามัธยฐาน ขณะที่โมเดลที่เทรนด้วยฟิลด์
ในฟอร์มอย่างเดียวได้ 0.7672 ซึ่งสูงกว่า การจำกัดขอบเขตจึงเพิ่มคุณภาพ
ไม่ใช่ลดคุณภาพ

### การเลือกฟีเจอร์

1. ใช้ VarianceThreshold ตัดคอลัมน์ค่าคงที่
2. บังคับเก็บฟีเจอร์ทุกตัวที่สืบย้อนไปถึงคอลัมน์ที่หน้าเว็บส่งมาได้
3. เติมฟีเจอร์ที่เหลือตามคะแนน ANOVA F-test
4. จำกัดจำนวนสุดท้ายไว้ที่ `--max-features` ค่าเริ่มต้น 200

หลังเลือกเสร็จ `verify_served_coverage` จะตรวจว่าทุกคอลัมน์ที่ฟอร์มเก็บมา
ยังเหลืออยู่ในโมเดล ไม่ว่าจะในรูปคอลัมน์ตรง missing indicator one-hot
หรือผ่านฟีเจอร์ที่ derive ออกมา หากมีคอลัมน์ใดหลุดหายไป การเทรนจะหยุด
พร้อมบอกชื่อคอลัมน์ เดิมระบบข้ามให้เงียบ ทำให้ช่องกรอกที่ผู้ใช้เห็น
อาจไม่มีผลต่อคำตอบโดยไม่มีใครรู้

## 9. การฝึก LightGBM

โมเดลหลักคือ LightGBMClassifier ซึ่งเหมาะกับข้อมูลตาราง
และรองรับความสัมพันธ์แบบไม่เป็นเส้นตรง

ค่าหลัก:

- estimators: 400
- learning rate: 0.03
- leaves: 31
- minimum child samples: 40
- row sampling: 0.85 พร้อม `subsample_freq = 1`
- column sampling: 0.85
- L1/L2 regularization
- random state: 42

`subsample_freq` สำคัญมาก LightGBM จะทำ row bagging ก็ต่อเมื่อค่านี้ตั้งแต่ 1
ขึ้นไป โค้ดเดิมตั้ง `subsample = 0.85` แต่ปล่อย `subsample_freq` เป็นค่าปริยาย 0
การสุ่มแถวจึงไม่เคยเกิดขึ้นจริงและไม่มีข้อความเตือนใดๆ

ฝึกด้วยข้อมูลเต็ม:

~~~powershell
python src/model_training.py
~~~

Smoke test ด้วยข้อมูลบางส่วน:

~~~powershell
python src/model_training.py --sample-rows 20000 --output models/dev_bundle.joblib
~~~

ห้ามนำ metric จาก smoke test ไปใช้อ้างอิงเป็นผลของโมเดลเต็ม

## 10. Probability Calibration และ Threshold

Probability ดิบจาก classifier อาจไม่ตรงกับอัตราผิดนัดจริง
ระบบจึงใช้ IsotonicRegression กับ calibration split

หลัง calibration ระบบเลือก threshold ที่ให้ F1 score สูงที่สุดบน threshold split
ซึ่งเป็นคนละชุดกับที่ใช้ fit calibrator แล้วนำ threshold ไปประเมินเพียงครั้งเดียว
บน validation split

ผลการฝึกล่าสุด scope `form` 22 คอลัมน์ 150 ฟีเจอร์:

| Metric | ค่า |
|---|---:|
| ROC-AUC | 0.7665 |
| Average precision | 0.2416 |
| Brier score | 0.0674 |
| Decision threshold | 0.1665 |
| Accuracy | 0.8536 |
| Precision | 0.2498 |
| Recall | 0.4060 |
| F1 | 0.3092 |
| Positive rate | 0.0807 |

ROC-AUC วัดความสามารถในการจัดลำดับความเสี่ยง ส่วน Brier score
วัดความคลาดเคลื่อนของ probability โดยค่ายิ่งต่ำยิ่งดี

Average precision เป็นตัวเลขที่ควรดูคู่กันเสมอ เพราะ positive rate อยู่ที่ 8%
ROC-AUC ในงาน imbalanced มักดูดีเกินจริง ส่วน average precision สะท้อน
คุณภาพการจัดอันดับเฉพาะกลุ่มที่ผิดนัดจริง เส้นฐานของมันคือ 0.0807
ซึ่งเท่ากับสัดส่วนผู้ผิดนัดในข้อมูล

ตัวเลขชุดนี้ต่ำกว่า 0.7756 ที่เคยรายงานไว้ แต่เทียบกันตรงๆ ไม่ได้ เพราะ 0.7756
มาจากโมเดล 495 คอลัมน์ที่ประเมินตอนมีข้อมูลครบทุกช่อง ซึ่งเป็นสถานการณ์ที่
หน้าเว็บไม่เคยมี เมื่อวัดโมเดลตัวนั้นในสภาพที่หน้าเว็บเสิร์ฟจริงจะได้ 0.7612
ต่ำกว่าโมเดลปัจจุบัน ตัวเลข 0.7665 จึงเป็นตัวเลขแรกของโปรเจกต์นี้ที่อธิบาย
ระบบที่ใช้งานจริงได้ตรงตามความเป็นจริง

## 11. Model Bundle

ผลการฝึกอยู่ใน models/credit_scoring_bundle.joblib ซึ่งประกอบด้วย:

- Bundle version และเวลาที่ฝึก
- รายชื่อ input columns และ served columns
- Default values จาก training split
- ตัวเลือกของแต่ละดรอปดาวน์ และอัตราการขาดหายของแต่ละช่อง
- Imputer, encoder และ scaler
- Variance filter
- Selected feature indices และ names
- LightGBM model
- Isotonic calibrator
- Probability bounds สำหรับบีบค่าไม่ให้แตะ 0 กับ 1
- Score scale คือจุดยึดกับ PDO ของดัชนี 300–850
- Decision และ risk thresholds
- Validation metrics
- Training metadata

ไฟล์ประกอบตั้งชื่อตาม stem ของ bundle:

- models/credit_scoring_bundle_metrics.json
- models/credit_scoring_bundle_feature_importance.csv

เดิมไฟล์สองตัวนี้ใช้ชื่อตายตัวว่า `model_metrics.json` และ
`feature_importance.csv` การรัน smoke test ด้วย `--output models/dev_bundle.joblib`
จึงเขียนทับผลของโมเดลเต็ม ซึ่งตรงกับสิ่งที่เอกสารนี้เตือนห้ามไว้พอดี
ปัจจุบันชื่อไฟล์ผูกกับ stem ของ bundle จึงแยกกันชัดเจน

การรวมองค์ประกอบใน bundle เดียวช่วยป้องกันการใช้ scaler,
feature order หรือ threshold ผิดรุ่น

### Probability bounds

Isotonic calibration ปล่อยค่า 0 กับ 1 พอดีออกมาได้ ในถังปลายสุดที่ข้อมูล
calibration ไม่มีคนผิดนัดเลยหรือผิดนัดทุกคน ปัญหาคือถังปลายสุดเล็กมาก

| ถัง | จำนวนแถว calibration | ผิดนัด |
|---|---:|---:|
| ระดับ 0% | 86 | 0 |
| ระดับ 100% | 4 | 4 |

ไม่มีขนาดตัวอย่างเท่านี้ที่รองรับข้อความว่าไม่มีทางผิดนัดหรือผิดนัดแน่นอน

การเลือกขอบเขตจากรูปร่างของเส้น calibration อย่างเดียวก็ไม่พอ รุ่นก่อนหน้า
ใช้กึ่งกลางระหว่างศูนย์กับระดับต่ำสุด ได้เพดานที่ 99.84% ซึ่งเกินหลักฐาน
ไปมาก เพราะ 370 แถวที่เสี่ยงที่สุดใน calibration ผิดนัดจริงเพียง 49%

ระบบปัจจุบันจึงรวมระดับจากปลายเข้าหากลางจนครบ 200 แถว แล้วใช้ค่า Jeffreys
ของกลุ่มที่รวมได้ ผลคือ

| ขอบเขต | ค่า | ที่มา |
|---|---:|---|
| พื้น | 0.5556% | รวม 2 ระดับล่างสุด 269 แถว ผิดนัด 1 คน |
| เพดาน | 49.19% | รวม 7 ระดับบนสุด 370 แถว ผิดนัด 182 คน |

ต้นทุนของการซื่อสัตย์นี้แทบเป็นศูนย์ ROC-AUC ลดจาก 0.7664901 เหลือ 0.7664883
หรือ 0.0000018 ส่วนผู้ที่ถูกตรึงที่ขอบมีเพียง 0.99% ที่พื้นและ 0.33% ที่เพดาน

ผลพลอยได้คือดัชนีไม่ชนขอบ 300 หรือ 850 อีกเลย ช่วงที่แสดงจริงคือ 427 ถึง 799
ทุกค่าในช่วงนี้มีความหมาย ไม่ใช่ค่าที่ถูกตัดปลาย

ขั้นตอนฝึกใช้การบีบค่าชุดเดียวกันก่อนเลือก threshold และคำนวณ metrics
ตัวเลขที่รายงานจึงอธิบายสิ่งที่หน้าเว็บเสิร์ฟจริง

### Score scale

ดัชนี 300–850 มีสองพารามิเตอร์ที่เก็บไว้ใน bundle และเลือกจากการวัด ไม่ใช่ธรรมเนียม

| พารามิเตอร์ | ค่า | เหตุผล |
|---|---|---|
| จุดยึด | 600 คะแนน ที่ odds 11.39 ต่อ 1 | เท่ากับอัตราผิดนัดของพอร์ตเอง 8.07% |
| PDO | 50 คะแนนต่อการทบเท่าของ odds | ให้การกระจายใช้สเกลได้จริง |

เดิมยึดที่ odds 50 ต่อ 1 ซึ่งเป็นค่าตายตัว ทำให้ผู้สมัครค่ากลางของพอร์ตได้ 557
บนสเกลที่คนอ่านเทียบกับ FICO เลขนี้แปลว่าเครดิตแย่ ทั้งที่เขาคือคนธรรมดา
ปัจจุบันค่ากลางอยู่ที่ 644 และคนที่ PD เท่ากับอัตราผิดนัดของพอร์ตพอดีได้ 600

เดิม PDO เท่ากับ 20 ตามธรรมเนียม scorecard ทำให้ช่วง p5 ถึง p95 กว้างเพียง
103 คะแนนจากสเกล 550 คะแนน และ 63.4% ของผู้สมัครกระจุกอยู่ในช่วง 550–600
ปัจจุบันช่วง p5 ถึง p95 กว้าง 258 คะแนน และช่วง 550–600 เหลือ 19.5%

ดัชนีนี้ผูกกับพอร์ตที่ใช้ฝึก ห้ามนำไปเทียบกับคะแนนเครดิตบูโรหรือเทียบข้ามพอร์ต

การกระจายบน validation หลังแก้:

| เปอร์เซ็นไทล์ | สกอร์ |
|---|---:|
| p5 | 493 |
| p25 | 584 |
| p50 | 644 |
| p75 | 679 |
| p95 | 751 |

และสกอร์ยังเรียงความเสี่ยงได้ถูกต้อง โดยค่าที่ทำนายใกล้ความจริงทุกช่วง

| ช่วงสกอร์ | จำนวน | ผิดนัดจริง | ทำนาย |
|---|---:|---:|---:|
| 400–499 | 1,817 | 31.92% | 33.61% |
| 500–549 | 2,590 | 19.11% | 18.48% |
| 550–599 | 6,010 | 11.35% | 10.67% |
| 600–649 | 7,318 | 5.77% | 5.77% |
| 650–699 | 8,251 | 2.92% | 3.01% |
| 700–799 | 4,766 | 1.32% | 1.16% |

## 11.1 ความแม่นยำของตัวเลขที่รายงาน

ตัวเลขทุกตัวมาจาก validation split เดียว 30,752 แถว จึงมีความไม่แน่นอน
ที่ต้องระบุ ค่าด้านล่างได้จาก bootstrap 2,000 รอบ

| Metric | ค่า | ช่วงเชื่อมั่น 95% |
|---|---:|---|
| ROC-AUC | 0.766 | 0.757 ถึง 0.776 |
| Average precision | 0.242 | 0.227 ถึง 0.258 |
| Brier score | 0.067 | 0.065 ถึง 0.070 |

**ห้ามรายงาน ROC-AUC เกินสองตำแหน่งทศนิยม** ความกว้างของช่วงเชื่อมั่นคือ
0.019 การเขียน 0.7665 จึงเป็นความแม่นยำเทียม ควรพูดว่าประมาณ 0.77

Decision threshold ก็ไม่นิ่ง bootstrap บน threshold split ให้ช่วง 0.1593
ถึง 0.1845 หรือกว้าง 15% ของตัวค่าเอง ผลกระทบต่อการใช้งานจริง

| threshold | precision | recall | F1 | สัดส่วนที่ถูกตีธง |
|---|---:|---:|---:|---:|
| 0.1593 | 0.244 | 0.433 | 0.312 | 14.3% |
| 0.1665 | 0.250 | 0.406 | 0.309 | 13.1% |
| 0.1845 | 0.278 | 0.348 | 0.309 | 10.1% |

F1 แทบไม่ขยับเพราะเป็นค่าที่ถูก optimize แต่ precision กับ recall
ขยับมาก และสัดส่วนผู้สมัครที่ถูกตีธงเปลี่ยนจาก 10.1% เป็น 14.3%
ซึ่งต่างกันในเชิงปฏิบัติการอย่างมีนัยสำคัญ

ก่อนนำ threshold ไปใช้จริงควรเลือกจากต้นทุนของ False Positive กับ
False Negative ไม่ใช่จากค่า F1 สูงสุด

## 12. ขั้นตอนทำนาย

ฟังก์ชัน inference อยู่ใน src/inference.py เมื่อรับข้อมูลผู้สมัคร ระบบจะ:

1. โหลด model bundle
2. ตรวจว่า key ที่ส่งมาอยู่ใน input schema ทุกตัว หากมี key แปลกปลอมจะ raise
3. สร้างหนึ่งแถวตาม input schema ตอนฝึก
4. คอลัมน์ใน `served_columns` ที่ไม่ได้ส่งมา ให้เป็นค่าว่าง ไม่ใช่มัธยฐาน
5. คอลัมน์นอก `served_columns` จึงจะใช้ default จาก training split
6. เรียกสูตร feature generation ชุดเดียวกับตอนฝึก
7. Transform ด้วย preprocessor
8. ใช้ variance filter และเลือกฟีเจอร์ตาม selected_indices
9. ทำนาย probability ดิบด้วย LightGBM
10. Calibrate probability
11. บีบค่าด้วย probability bounds ไม่ให้แตะ 0 กับ 1
12. แปลง probability เป็นระดับความเสี่ยงและดัชนี 300–850 ตาม score scale ใน bundle

ข้อ 4 คือหัวใจ การเติมมัธยฐานให้ช่องที่ไม่มีข้อมูลจะทำให้ missing indicator
ของ imputer เป็น 0 ซึ่งบอกโมเดลว่า "มีข้อมูลและค่าเท่ากับค่ากลาง" ทั้งที่
ความจริงคือไม่มีข้อมูล ตอนเทรนโมเดลเห็นสถานะไม่มีข้อมูลอยู่บ่อยมาก
เช่น EXT_SOURCE_1 ขาดหายถึง 56% การส่งค่าว่างจริงจึงเป็นการรักษาความหมาย
เดียวกันระหว่างตอนเทรนกับตอนใช้งาน

หากภายหลังเปลี่ยนไปใช้ scope `application` หรือ `full` ฟีเจอร์ที่หน้าเว็บ
ไม่ได้รับจะกลับไปใช้ training default อีกครั้ง ซึ่งต้องเปลี่ยนเป็นการต่อ
ข้อมูลจริงจากแหล่งที่ได้รับอนุญาตก่อนนำขึ้น production

## 13. การทำงานของ Streamlit

หน้าเว็บอยู่ใน app/streamlit_app.py แบบฟอร์มรับ:

- ข้อมูลส่วนตัวและครอบครัว
- รายได้ วงเงิน ยอดผ่อน และราคาสินค้า
- การทำงาน อาชีพ และทรัพย์สิน
- คะแนนเครดิตภายนอกที่เจ้าหน้าที่กรอกจากระบบ

เมื่อกด **ประเมินความเสี่ยง**:

1. แปลงค่าจากแบบฟอร์มเป็นชื่อคอลัมน์ Home Credit
2. ตรวจว่าชุด key ตรงกับ `served_columns` ของ bundle หากไม่ตรงจะหยุดพร้อมบอกส่วนต่าง
3. เรียก predict_default_probability
4. แสดง calibrated probability
5. แสดงระดับความเสี่ยงตาม threshold ใน bundle
6. แสดงดัชนีแบบ odds-based

### คะแนนเครดิตภายนอก

ช่อง EXT_SOURCE ทั้งสามเป็นค่าที่เจ้าหน้าที่สินเชื่อดึงจากระบบเครดิตภายนอก
ไม่ใช่ค่าที่ผู้สมัครประเมินตัวเอง แต่ละช่องมีตัวเลือก **ไม่พบในระบบ**
ซึ่งจะส่งค่าว่างเข้าแบบจำลอง

ตัวเลือกนี้จำเป็น ไม่ใช่ของประดับ ในข้อมูลฝึก EXT_SOURCE_1 ขาดหายถึง 56%
หากบังคับให้มีค่าเสมอ โมเดลจะเห็นสถานะที่ต่างจากตอนเทรนในทุกคำทำนาย
เดิมช่องเหล่านี้เป็นสไลเดอร์ที่ตั้งค่าเริ่มต้นไว้ที่ 0.5 ซึ่งเท่ากับให้ผู้ใช้
เดาตัวแปรที่มีอิทธิพลสูงสุดต่อคำตอบ วัดแล้วการเลื่อนสไลเดอร์สามตัวนี้
ทำให้ probability เปลี่ยนจาก 0.006 ถึง 0.45 ขณะที่รายได้ต่อปีตั้งแต่ 30,000
ถึง 5,000,000 เปลี่ยน probability ไม่ถึง 0.01

### สถานะการจ้างงาน

หากเลือกประเภทรายได้เป็นผู้รับบำนาญหรือไม่ได้ทำงาน ระบบจะส่ง DAYS_EMPLOYED
เป็นค่าว่างและตั้ง DAYS_EMPLOYED_ANOM เป็น 1 พร้อมปิดช่องอายุงาน
ตรงกับข้อมูลฝึกซึ่งสองประเภทนี้มี sentinel 365243 ครบ 100% และไม่พบ
ในประเภทอื่นเลย เดิมหน้าเว็บส่งอายุงาน 0 ปีพร้อม ANOM เป็น 0
ซึ่งเป็นสถานะที่ข้อมูลฝึกไม่มีอยู่จริง

~~~powershell
python -m streamlit run app/streamlit_app.py
~~~

เปิดที่ http://localhost:8501

## 14. Tests

Tests อยู่ในโฟลเดอร์ tests และครอบคลุม:

- สูตร financial, time และ external-source features
- การป้องกัน division by zero และ infinity
- การแทน sentinel ของ DAYS_EMPLOYED
- การ aggregate ตารางประวัติ
- การสร้าง input row จาก defaults และการส่งค่าว่างสำหรับช่องที่ไม่ได้กรอก
- การ raise เมื่อพบ key ที่ไม่อยู่ใน schema
- การจำกัดขอบเขตฟีเจอร์ตาม scope
- การตรวจว่าทุกช่องกรอกยังเหลืออยู่ในโมเดล
- ความเป็น monotonic และขอบเขตของดัชนี 300–850
- การจัดระดับความเสี่ยง
- สัญญาระหว่างฟอร์มกับ bundle แบบ end-to-end ใน tests/test_serving_contract.py

~~~powershell
pytest
~~~

`pyproject.toml` ตั้ง `pythonpath` ไว้แล้ว จึงสั่ง `pytest` ตรงๆ ได้
เดิมต้องใช้ `python -m pytest` เท่านั้น เพราะไม่มีไฟล์ตั้งค่าใดๆ
ทำให้ `pytest` เปล่า collection error ทั้งหมด

ผลล่าสุด: **32 passed**

เทสต์ใน tests/test_serving_contract.py ต้องใช้ model bundle หากยังไม่ได้เทรน
เทสต์กลุ่มนี้จะถูก skip โดยไม่ทำให้ชุดทดสอบล้มเหลว

## 15. ลำดับคำสั่งตั้งแต่เริ่มต้น

~~~powershell
# 1. สร้าง environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. ติดตั้ง dependencies
python -m pip install -r requirements.txt

# 3. เตรียมข้อมูล
python src/data_preprocessing.py --reuse-aggregates

# 4. ตรวจคุณภาพ
python src/check_data_quality.py

# 5. ฝึกและประเมินโมเดล
python src/model_training.py

# 6. รัน tests
pytest

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
