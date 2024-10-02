import sqlite3
import pandas as pd


# df1 = pd.read_excel("./data/bvr-eat.xlsx")
# df2 = pd.read_excel("./data/cc-eat.xlsx")

df1 = pd.read_excel("./data/bvr-visit.xlsx")
df2 = pd.read_excel("./data/cc-visit.xlsx")

required_columns = ['NOME',	'INFORMACÕES',	'ENDEREÇO',	'LOCALIZAÇÃO',	'CONTATO',	'REDES SOCIAIS'
                    ]  # Add the columns you need
for df in [df1, df2]:
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # or pd.NA if you prefer the Pandas "missing" type


combined_df = pd.concat([df1, df2], ignore_index=True)

conn = sqlite3.connect('database.db')

# combined_df.to_sql('places_to_eat', conn, if_exists='replace', index=False)
combined_df.to_sql('places_to_visit', conn, if_exists='replace', index=False)

conn.commit()

conn.close()

print("Done!")
