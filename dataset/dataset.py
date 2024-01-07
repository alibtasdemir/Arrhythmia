import pandas
import torch
import numpy as np
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, df: pandas.DataFrame):
        self.dataframe = df
        self.data_columns = self.dataframe.columns[:-2].tolist()

    def __getitem__(self, idx):
        signal = self.dataframe.loc[idx, self.data_columns].astype('float32')
        signal = torch.from_numpy(signal.values).float()
        # signal = torch.FloatTensor([signal.values])
        target = torch.LongTensor(np.array(self.dataframe.loc[idx, 'class']))
        return signal, target

    def __len__(self):
        return len(self.dataframe)


"""
-- Complete attribute documentation:
      1 Age: Age in years , linear
      2 Sex: Sex (0 = male; 1 = female) , nominal
      3 Height: Height in centimeters , linear
      4 Weight: Weight in kilograms , linear
      5 QRS duration: Average of QRS duration in msec., linear
      6 P-R interval: Average duration between onset of P and Q waves
        in msec., linear
      7 Q-T interval: Average duration between onset of Q and offset
        of T waves in msec., linear
      8 T interval: Average duration of T wave in msec., linear
      9 P interval: Average duration of P wave in msec., linear
     Vector angles in degrees on front plane of:, linear
     10 QRS
     11 T
     12 P
     13 QRST
     14 J

     15 Heart rate: Number of heart beats per minute ,linear

     Of channel DI:
      Average width, in msec., of: linear
      16 Q wave
      17 R wave
      18 S wave
      19 R' wave, small peak just after R
      20 S' wave

      21 Number of intrinsic deflections, linear

      22 Existence of ragged R wave, nominal
      23 Existence of diphasic derivation of R wave, nominal
      24 Existence of ragged P wave, nominal
      25 Existence of diphasic derivation of P wave, nominal
      26 Existence of ragged T wave, nominal
      27 Existence of diphasic derivation of T wave, nominal

     Of channel DII: 
      28 .. 39 (similar to 16 .. 27 of channel DI)
     Of channels DIII:
      40 .. 51
     Of channel AVR:
      52 .. 63
     Of channel AVL:
      64 .. 75
     Of channel AVF:
      76 .. 87
     Of channel V1:
      88 .. 99
     Of channel V2:
      100 .. 111
     Of channel V3:
      112 .. 123
     Of channel V4:
      124 .. 135
     Of channel V5:
      136 .. 147
     Of channel V6:
      148 .. 159

     Of channel DI:
      Amplitude , * 0.1 milivolt, of
      160 JJ wave, linear
      161 Q wave, linear
      162 R wave, linear
      163 S wave, linear
      164 R' wave, linear
      165 S' wave, linear
      166 P wave, linear
      167 T wave, linear

      168 QRSA , Sum of areas of all segments divided by 10,
          ( Area= width * height / 2 ), linear
      169 QRSTA = QRSA + 0.5 * width of T wave * 0.1 * height of T
          wave. (If T is diphasic then the bigger segment is
          considered), linear

     Of channel DII:
      170 .. 179
     Of channel DIII:
      180 .. 189
     Of channel AVR:
      190 .. 199
     Of channel AVL:
      200 .. 209
     Of channel AVF:
      210 .. 219
     Of channel V1:
      220 .. 229
     Of channel V2:
      230 .. 239
     Of channel V3:
      240 .. 249
     Of channel V4:
      250 .. 259
     Of channel V5:
      260 .. 269
     Of channel V6:
      270 .. 279
"""

data_columns = [
    "age",
    "sex",
    "height",
    "weight",
    "QRS_duration",
    "PR_interval",
    "QT_interval",
    "T_interval",
    "P_interval",
    "QRS_angle",
    "T_angle",
    "P_angle",
    "QRST_angle",
    "J_angle",
    "heart_rate",
    "Q_wave_avgw",
    "R_wave_avgw",
    "S_wave_avgw",
    "Rp_wave_avgw",
    "Sp_wave_avgw",
    "no_int_deflections",
    "ragged_R_wave",
    "diphasic_R_wave",
    "ragged_P_wave",
    "diphasic_P_wave"
    "ragged_T_wave",
    "diphasic_T_wave",
    "DII_1",
    "DII_2",
    "DII_3",
    "DII_4",
    "DII_5",
    "DII_6",
    "DII_7",
    "DII_8",
    "DII_9",
    "DII_10",
    "DII_11",
    "DIII_1",
    "DIII_2",
    "DIII_3",
    "DIII_4",
    "DIII_5",
    "DIII_6",
    "DIII_7",
    "DIII_8",
    "DIII_9",
    "DIII_10",
    "DIII_11",
    "AVR_1",
    "AVR_2",
    "AVR_3",
    "AVR_4",
    "AVR_5",
    "AVR_6",
    "AVR_7",
    "AVR_8",
    "AVR_9",
    "AVR_10",
    "AVR_11",
    "AVL_1",
    "AVL_2",
    "AVL_3",
    "AVL_4",
    "AVL_5",
    "AVL_6",
    "AVL_7",
    "AVL_8",
    "AVL_9",
    "AVL_10",
    "AVL_11",
    "AVF_1",
    "AVF_2",
    "AVF_3",
    "AVF_4",
    "AVF_5",
    "AVF_6",
    "AVF_7",
    "AVF_8",
    "AVF_9",
    "AVF_10",
    "AVF_11",
    "V1_1",
    "V1_2",
    "V1_3",
    "V1_4",
    "V1_5",
    "V1_6",
    "V1_7",
    "V1_8",
    "V1_9",
    "V1_10",
    "V1_11",
    "V2_1",
    "V2_2",
    "V2_3",
    "V2_4",
    "V2_5",
    "V2_6",
    "V2_7",
    "V2_8",
    "V2_9",
    "V2_10",
    "V2_11",
    "V3_1",
    "V3_2",
    "V3_3",
    "V3_4",
    "V3_5",
    "V3_6",
    "V3_7",
    "V3_8",
    "V3_9",
    "V3_10",
    "V3_11",
    "V4_1",
    "V4_2",
    "V4_3",
    "V4_4",
    "V4_5",
    "V4_6",
    "V4_7",
    "V4_8",
    "V4_9",
    "V4_10",
    "V4_11",
    "V5_1",
    "V5_2",
    "V5_3",
    "V5_4",
    "V5_5",
    "V5_6",
    "V5_7",
    "V5_8",
    "V5_9",
    "V5_10",
    "V5_11",
    "V6_1",
    "V6_2",
    "V6_3",
    "V6_4",
    "V6_5",
    "V6_6",
    "V6_7",
    "V6_8",
    "V6_9",
    "V6_10",
    "V6_11",
    "JJ_wave_amp",
    "Q_wave_amp",
    "R_wave_amp",
    "S_wave_amp",
    "Rp_wave_amp",
    "Sp_wave_amp",
    "P_wave_amp",
    "T_wave_amp",
    "QRSA",
    "QRSTA",

]
