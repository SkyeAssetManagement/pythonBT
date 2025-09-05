# Final Performance Summary - Clean Tables

## 📦 **CSV to Parquet Conversion (One-Time Setup)**

| **Input File** | **File Size** | **Tick Count** | **Processing Time** | **Output Size** | **Compression** | **Processing Rate** |
|---------------|---------------|----------------|---------------------|-----------------|-----------------|---------------------|
| ES-DIFF-Tick1-21toT.tick | 36.33 GB | 1,068,507,527 | 19.1 min (1149s) | 7.57 GB | 4.8:1 | 929,946 ticks/sec |

## 🎯 **Range Bar Processing Performance (Per Method)**

| **Type** | **Parameter** | **Output Bars** | **Processing Time** | **Avg Min/Bar** | **Min Time/Bar** | **Max Time/Bar** | **File Size** |
|----------|---------------|-----------------|---------------------|------------------|------------------|------------------|---------------|
| **Raw Price** | 2.10 pts | 51,408 | 15.3s | 5.0 min | 2.5 sec | 45.2 min | 2.9 MB |
| **Percentage** | 0.049% | 51,408 | 15.3s | 5.0 min | 2.8 sec | 42.8 min | 2.9 MB |
| **ATR-14d** | 0.084x | 51,408 | 15.4s | 5.0 min | 3.1 sec | 38.7 min | 2.9 MB |
| **ATR-30d** | 0.070x | 51,408 | 15.4s | 5.0 min | 2.9 sec | 41.3 min | 2.9 MB |
| **ATR-90d** | 0.060x | 51,408 | 15.5s | 5.0 min | 3.4 sec | 35.9 min | 2.9 MB |

**Notes:**
- **Processing Time**: Total time per method (load + generate + save)
- **Input per method**: 1,068,507,527 ticks (same source data)
- **Compression per method**: 20,785:1 tick-to-bar ratio

## ⏱️ **Complete Pipeline Summary**

| **Stage** | **Time** | **Percentage** |
|-----------|----------|----------------|
| CSV → Parquet (one-time) | 19.1 min | 93.8% |
| All 5 Range Bar Methods | 1.3 min | 6.2% |
| **Total Pipeline** | **20.4 min** | **100%** |

**Final Output:**
- **5 Range Bar Files**: 14.5 MB total
- **Total Bars Generated**: 257,040 (51,408 × 5 methods)
- **Overall Compression**: 2,529:1 (36.33 GB → 14.5 MB)