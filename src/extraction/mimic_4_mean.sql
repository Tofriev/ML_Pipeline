WITH weight_data AS (-- first select all weight data 
    SELECT
        icu.stay_id,
        icu.intime,
        ce.charttime,
        ce.valuenum AS weight,
        CASE
            WHEN ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') THEN 'after'
            ELSE 'before'
        END AS time_category
    FROM
        icustays icu
        INNER JOIN chartevents ce ON icu.stay_id = ce.stay_id
    WHERE
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (224639, 226512) --d aily weight and admission weight 
        AND ce.valuenum <= 500
        AND ce.valuenum >= 20
),
weight_after AS ( -- then take avg of first 24 h after intime
    SELECT
        stay_id,
        AVG(weight) AS weight_mean
    FROM weight_data
    WHERE time_category = 'after'
    GROUP BY stay_id
),
weight_before AS ( -- get closest weght from before 
    SELECT
        stay_id,
        weight AS weight_mean,
        ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY ABS(JULIANDAY(charttime) - JULIANDAY(intime))) AS rn
    FROM weight_data
    WHERE time_category = 'before'
),
weight_final AS ( -- finally display weight after intime if avaliable otherwise take the first from before (marked by rn)
    SELECT
        stay_id,
        weight_mean
    FROM weight_after
    UNION ALL
    SELECT
        stay_id,
        weight_mean
    FROM weight_before
    WHERE rn = 1 AND stay_id NOT IN (SELECT stay_id FROM weight_after)
),
height_data AS ( -- same logic for height 
    SELECT
        icu.stay_id,
        icu.intime,
        ce.charttime,
        ce.valuenum AS height,
        CASE
            WHEN ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') THEN 'after'
            ELSE 'before'
        END AS time_category
    FROM
        icustays icu
        INNER JOIN chartevents ce ON icu.stay_id = ce.stay_id
    WHERE
        ce.valuenum IS NOT NULL AND
        ce.valuenum != 0 AND
        ce.itemid IN (226730) 
        AND ce.valuenum <= 260
),
height_after AS (
    SELECT
        stay_id,
        AVG(height) AS height_mean
    FROM height_data
    WHERE time_category = 'after'
    GROUP BY stay_id
),
height_before AS (
    SELECT
        stay_id,
        height AS height_mean,
        ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY ABS(JULIANDAY(charttime) - JULIANDAY(intime))) AS rn
    FROM height_data
    WHERE time_category = 'before'
),
height_final AS (
    SELECT
        stay_id,
        height_mean
    FROM height_after
    UNION ALL
    SELECT
        stay_id,
        height_mean
    FROM height_before
    WHERE rn = 1 AND stay_id NOT IN (SELECT stay_id FROM height_after)
), 
temperature AS (
  SELECT
        ce.stay_id,
        AVG(VALUENUM) as temperature_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (223762) 
        AND ce.valuenum <= 45
        AND ce.valuenum >= 20
    GROUP BY
        ce.stay_id
), 
resprate AS (
  SELECT
        ce.stay_id,
        AVG(VALUENUM) as rr_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220210) 
        AND ce.valuenum <= 50
        AND ce.valuenum >= 5
    GROUP BY
        ce.stay_id
), 
heartrate AS (
  SELECT
        ce.stay_id,
        AVG(VALUENUM) as hr_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220045) 
        AND ce.valuenum <= 300
        AND ce.valuenum >= 10
    GROUP BY
        ce.stay_id
), 
glucose AS (
  SELECT
        ce.stay_id,
        AVG(VALUENUM) as glc_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.valuenum != 0 AND
        ce.itemid IN (220621) -- serum glc
        AND ce.valuenum <= 2000
        AND ce.valuenum >= 5
    GROUP BY
        ce.stay_id
), 
systolic_bp AS (
    SELECT
        ce.stay_id,
        AVG(VALUENUM) as sbp_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.valuenum != 0 AND
        ce.itemid IN (220050, 220179, 224167, 227243) 
        AND ce.valuenum <= 400
    GROUP BY
        ce.stay_id
), 
diastolic_bp AS (
    SELECT
        ce.stay_id,
        AVG(VALUENUM) as dbp_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220051, 220180, 224643, 227242) 
        AND ce.valuenum <= 350
        AND ce.valuenum >= 20 
    GROUP BY
        ce.stay_id
), 
mean_bp AS (
    SELECT
        ce.stay_id,
        AVG(VALUENUM) as mbp_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.valuenum != 0 AND
        ce.itemid IN (220052, 220181) 
        AND ce.valuenum <= 400
        AND ce.valuenum >= 20
    GROUP BY
        ce.stay_id
), 
ph AS (
    SELECT
        ce.stay_id,
        AVG(VALUENUM) as ph_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220274, 223830) --venous, arterial 
        AND ce.valuenum <= 9
        AND ce.valuenum >= 5
    GROUP BY
        ce.stay_id
), 
mort AS ( -- only death during icu stay is recorded
    SELECT
        ic.subject_id,
        ic.stay_id,
        ic.hadm_id,
        CASE
            WHEN adm.deathtime BETWEEN ic.intime AND ic.outtime THEN 1
            WHEN adm.dischtime <= ic.outtime AND adm.discharge_location = 'DIED' THEN 1 
            WHEN adm.deathtime <= ic.intime THEN 1
            ELSE 0
        END AS Mortality_icu    
    FROM icustays ic
    INNER JOIN admissions adm ON ic.hadm_id = adm.hadm_id
),
gcs_eyes AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS mean_eyes
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL 
        AND ce.valuenum <= 4
        AND ce.valuenum >= 1
        AND ce.itemid IN (220739) 
    GROUP BY
        ce.stay_id
), 
gcs_verbal AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS mean_verbal 
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (223900) 
        AND ce.valuenum <= 5
        AND ce.valuenum >= 1
    GROUP BY
        ce.stay_id
)
, gcs_motor AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS mean_motor 
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (223901) 
        AND ce.valuenum <= 6
        AND ce.valuenum >= 1
    GROUP BY
        ce.stay_id
)
, pao2 AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS pao2_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220224)
        AND ce.valuenum <= 300
        AND ce.valuenum >= 10
    GROUP BY
        ce.stay_id
)
, creatinine_serum AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS creatinine_serum_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220615)
        AND ce.valuenum <= 20
        AND ce.valuenum >= 0.1
    GROUP BY
        ce.stay_id
)
, fio2_normal AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS fio2_normal_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (223835)
        AND ce.valuenum <= 100
        AND ce.valuenum >= 20
    GROUP BY
            ce.stay_id
)
, potassium_serum AS ( -- Kalium 
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS potassium_serum_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (227442)
        AND ce.valuenum <= 7
        AND ce.valuenum >= 2.5
    GROUP BY
            ce.stay_id
)
, sodium_serum AS (-- Natrium 
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS sodium_serum_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220645)
        AND ce.valuenum <= 160
        AND ce.valuenum >= 120
    GROUP BY
            ce.stay_id
)
, wbc AS ( --leucocyten
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS wbc_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220546)
        AND ce.valuenum <= 200
        AND ce.valuenum >= 1
    GROUP BY
            ce.stay_id
)
, platelets AS ( --thombocyten
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS platelets_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (227457)
        AND ce.valuenum <= 1000
        AND ce.valuenum >= 10
    GROUP BY
            ce.stay_id
)
, bilirubin_total AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS bilirubin_total_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (225690)
        AND ce.valuenum <= 50
        AND ce.valuenum >= 0.1
    GROUP BY
        ce.stay_id
)
, hco3_serum AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS hco3_serum_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (227443)
        AND ce.valuenum <= 45
        AND ce.valuenum >= 10
    GROUP BY
            ce.stay_id
)
, haemoglobin AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS haemoglobin_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220228)
        AND ce.valuenum <= 20
        AND ce.valuenum >= 1
        
    GROUP BY
            ce.stay_id
)
, inr AS ( --ZINR/Quick
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS inr_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (227467)
        AND ce.valuenum <= 6
        AND ce.valuenum >= 0.2
    GROUP BY
            ce.stay_id
)
, alat AS ( 
SELECT 
    ce.stay_id,
    AVG(VALUENUM) AS alat_mean
FROM
    chartevents ce
INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
WHERE
    ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
    ce.valuenum IS NOT NULL AND
    ce.itemid IN (220644)
    AND ce.valuenum <= 2000
    AND ce.valuenum >= 2
GROUP BY

        ce.stay_id
)
, asat AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS asat_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220587)
        AND ce.valuenum <= 2000
        AND ce.valuenum >= 2
    GROUP BY
            ce.stay_id
)
, paco2 AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS paco2_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (220235)
        AND ce.valuenum <= 300
        AND ce.valuenum >= 10
    GROUP BY
            ce.stay_id
)
, albumin AS  (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS albumin_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (227456)
        AND ce.valuenum <= 60
        AND ce.valuenum >= 2
    GROUP BY
            ce.stay_id
)
, anion_gap AS (
    SELECT 
        ce.stay_id,
        AVG(VALUENUM) AS anion_gap_mean
    FROM
        chartevents ce
    INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
    WHERE
        ce.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') AND
        ce.valuenum IS NOT NULL AND
        ce.itemid IN (227073)
        AND ce.valuenum <= 25
        AND ce.valuenum >= 1
    GROUP BY
            ce.stay_id
)
, lactate AS ( 
    SELECT
        icu.stay_id,
        AVG(l.valuenum) AS lactate_mean
    FROM
        labevents l
    INNER JOIN icustays icu ON l.hadm_id = icu.hadm_id
    WHERE
        l.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') 
        AND l.valuenum IS NOT NULL
        AND l.itemid IN (50813, 52442, 53154)
        AND l.valuenum <= 200
        AND l.valuenum >= 0.1
    GROUP BY
        icu.hadm_id,
        icu.stay_id
)
, urea_nitrogen_blood AS (
    SELECT
        icu.stay_id,
        AVG(l.valuenum) AS urea_nitrogen_blood_mean
    FROM
        labevents l
    INNER JOIN icustays icu ON l.hadm_id = icu.hadm_id
    WHERE
        l.charttime BETWEEN icu.intime AND datetime(icu.intime, '+24 hours') 
        AND l.valuenum IS NOT NULL
        AND l.itemid IN (52647, 51006)
    GROUP BY
        icu.hadm_id,
        icu.stay_id
)

            


SELECT
    m.Mortality_icu AS mortality,
    CAST((strftime('%s', ic.outtime) - strftime('%s', ic.intime)) / 3600.0 AS INTEGER) AS LOS, -- LOS in hours
    CASE 
        WHEN adm.race IN ('ASIAN', 'ASIAN - KOREAN', 'ASIAN - CHINESE', 'ASIAN - SOUTH EAST ASIAN', 'ASIAN - ASIAN INDIAN') THEN 1
        WHEN adm.race IN ('BLACK/AFRICAN AMERICAN', 'BLACK/CARIBBEAN ISLAND', 'BLACK/AFRICAN','BLACK/CAPE VERDEAN', 'CARIBBEAN ISLAND') THEN 2
        WHEN adm.race IN ('HISPANIC', 'HISPANIC/LATINO - CENTRAL AMERICAN', 'HISPANIC/LATINO - COLUMBIAN', 'HISPANIC/LATINO - HONDURAN', 'HISPANIC/LATINO - CUBAN', 'HISPANIC/LATINO - MEXICAN', 'HISPANIC OR LATINO', 'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - SALVADORAN', 'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - GUATEMALAN', 'SOUTH AMERICAN') THEN 3
        WHEN adm.race IN ('WHITE', 'WHITE - BRAZILIAN', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN', 'MIDDLE EASTERN', 'PORTUGUESE') THEN 4
        WHEN adm.race IN ('AMERICAN INDIAN', 'AMERICAN INDIAN/ALASKA NATIVE', 'NATIVE HAWAIIAN','NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'MULTIPLE RACE/ETHNICITY', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER', 'UNKNOWN', 'OTHER', '') THEN 0
    END AS Eth,
    p.gender,
    p.anchor_age AS Age,
    wf.weight_mean AS "Weight",
    hf.height_mean AS "Height",
    (wf.weight_mean / ((hf.height_mean / 100) * (hf.height_mean / 100))) AS "Bmi", -- convert height from cm in m for bmi calculation
    t.temperature_mean AS Temp,
    r.rr_mean AS "RR",
    hr.hr_mean AS "HR",
    g.glc_mean AS "GLU"
    , sbp.sbp_mean AS "SBP"
    , dbp.dbp_mean AS "DBP"
    , mbp.mbp_mean AS "MBP"
    , ph.ph_mean AS "Ph"
    , gcs_e.mean_eyes + gcs_v.mean_verbal + gcs_m.mean_motor AS "GCST"
    , pao2.pao2_mean AS "PaO2"
    , creatinine_serum.creatinine_serum_mean AS "Kreatinin"
    , fio2_normal.fio2_normal_mean AS "FiO2"
    , potassium_serum.potassium_serum_mean AS "Kalium"
    , sodium_serum.sodium_serum_mean AS "Natrium"
    , wbc.wbc_mean AS "Leukocyten"
    , platelets.platelets_mean AS "Thrombocyten"
    , bilirubin_total.bilirubin_total_mean AS "Bilirubin"
    , hco3_serum.hco3_serum_mean AS "HCO3"
    , haemoglobin.haemoglobin_mean AS "Hb"
    , inr.inr_mean AS "Quick"
    , alat.alat_mean AS "ALAT"
    , asat.asat_mean AS "ASAT"
    , paco2.paco2_mean AS "PaCO2"
    , albumin.albumin_mean AS "Albumin"
    , anion_gap.anion_gap_mean AS "AnionGAP"
    , lactate.lactate_mean AS "Lactate"
    , urea_nitrogen_blood.urea_nitrogen_blood_mean AS "Harnstoff"
    

FROM
    icustays ic
INNER JOIN admissions adm ON ic.hadm_id = adm.hadm_id
INNER JOIN patients p ON ic.subject_id = p.subject_id
LEFT JOIN weight_final AS wf ON ic.stay_id = wf.stay_id
LEFT JOIN height_final AS hf ON ic.stay_id = hf.stay_id
LEFT JOIN temperature AS t ON 1=1
    AND ic.stay_id = t.stay_id
LEFT JOIN resprate AS r ON 1=1
    AND ic.stay_id = r.stay_id
LEFT JOIN heartrate AS hr ON 1=1
    AND ic.stay_id = hr.stay_id
LEFT JOIN glucose AS g ON 1=1
    AND ic.stay_id = g.stay_id
LEFT JOIN systolic_bp AS sbp ON 1=1
    AND ic.stay_id = sbp.stay_id
LEFT JOIN diastolic_bp AS dbp ON 1=1
    AND ic.stay_id = dbp.stay_id 
LEFT JOIN mean_bp AS mbp ON 1=1
    AND ic.stay_id = mbp.stay_id  
LEFT JOIN mort AS m ON ic.stay_id = m.stay_id  
LEFT JOIN ph AS ph ON ic.stay_id = ph.stay_id  
LEFT JOIN gcs_eyes AS gcs_e ON ic.stay_id = gcs_e.stay_id 
LEFT JOIN gcs_verbal AS gcs_v ON ic.stay_id = gcs_v.stay_id 
LEFT JOIN gcs_motor AS gcs_m ON ic.stay_id = gcs_m.stay_id 
LEFT JOIN pao2 AS pao2 ON ic.stay_id = pao2.stay_id 
LEFT JOIN creatinine_serum ON ic.stay_id = creatinine_serum.stay_id
LEFT JOIN fio2_normal ON ic.stay_id = fio2_normal.stay_id
LEFT JOIN potassium_serum ON ic.stay_id = potassium_serum.stay_id
LEFT JOIN sodium_serum ON ic.stay_id = sodium_serum.stay_id
LEFT JOIN wbc ON ic.stay_id = wbc.stay_id
LEFT JOIN platelets ON ic.stay_id = platelets.stay_id
LEFT JOIN bilirubin_total ON ic.stay_id = bilirubin_total.stay_id
LEFT JOIN hco3_serum ON ic.stay_id = hco3_serum.stay_id
LEFT JOIN haemoglobin ON ic.stay_id = haemoglobin.stay_id
LEFT JOIN inr ON ic.stay_id = inr.stay_id
LEFT JOIN alat ON ic.stay_id = alat.stay_id 
LEFT JOIN asat ON ic.stay_id = asat.stay_id
LEFT JOIN paco2 ON ic.stay_id = paco2.stay_id
LEFT JOIN albumin ON ic.stay_id = albumin.stay_id
LEFT JOIN anion_gap ON ic.stay_id = anion_gap.stay_id
LEFT JOIN lactate ON ic.stay_id = lactate.stay_id
LEFT JOIN urea_nitrogen_blood ON ic.stay_id = urea_nitrogen_blood.stay_id

    
     