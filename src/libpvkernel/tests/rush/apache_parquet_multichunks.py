#!/usr/bin/env python

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def write_parquet_in_chunks(df, chunk_size, file_name):
    writer = None
    for start in range(0, len(df), chunk_size):
        end = start + chunk_size
        chunk = df.iloc[start:end]
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(file_name, table.schema)
        writer.write_table(table)
    if writer:
        writer.close()

format='%Y-%m-%d %H:%M:%S'
data = {
    'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'col2': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    'col3': [b'\x01', b'\x02', b'\x03', b'\x04', b'\x05', b'\x06', b'\x07', b'\x08', b'\x09', b'\x0A'],
    'col4': [
                pd.to_datetime('2018-10-26 12:00:00', format=format),
                pd.to_datetime('2019-10-26 12:00:00', format=format),
                pd.to_datetime('2020-10-26 12:00:00', format=format),
                pd.to_datetime('2021-10-26 12:00:00', format=format),
                pd.to_datetime('2022-10-26 12:00:00', format=format),
                pd.to_datetime('2023-10-26 12:00:00', format=format),
                pd.to_datetime('2024-10-26 12:00:00', format=format),
                pd.to_datetime('2025-10-26 12:00:00', format=format),
                pd.to_datetime('2026-10-26 12:00:00', format=format),
                pd.to_datetime('2027-10-26 12:00:00', format=format)
            ]
}
df = pd.DataFrame(data)

write_parquet_in_chunks(df, 3, 'multichunks.parquet')