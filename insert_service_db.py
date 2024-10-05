import sqlite3
import pandas as pd

sheet_name = 'visit'
df1 = pd.read_excel(f"./data/{sheet_name}.xlsx")


conn = sqlite3.connect('database.db')

# combined_df.to_sql('places_to_eat', conn, if_exists='replace', index=False)
df1.to_sql(f'places_to_{sheet_name}', conn, if_exists='replace', index=False)

conn.commit()

conn.close()

print("Done!")
