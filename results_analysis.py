import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##### LIME results

with open('results_lime.pkl', 'rb') as f:
    results = pkl.load(f)
    f.close()

rows = []
cumulative_road = {}
for entry in results:
    id_ = entry[0]
    label = entry[1]
    road_dict = entry[2]['ROAD']
    complexity_score = entry[2]['COMPLEXITY'][0]
    faithfulness_score = entry[2]['FAITHFULNESS'][0]
    jaccard_set = entry[3]
    
    rows.append({
        'id': id_,
        'label': label,
        'complexity': complexity_score,
        'road': road_dict,
        'best jaccard': jaccard_set[0],
        'overlap ratio': jaccard_set[2],
        'faithfulness_score' : faithfulness_score
    })

df = pd.DataFrame(rows)

# Extraction et moyenne des valeurs ROAD
road_keys = list(df.iloc[0]['road'].keys())
road_matrix = pd.DataFrame([d['road'] for d in rows])

average_road : dict = road_matrix.mean().to_dict()
average_complexity = df['complexity'].mean()
average_jaccard = df['best jaccard'].mean()
average_faithfulness = df['faithfulness_score'].mean()

# Affichage
print("DataFrame:")
print(df[['id', 'label', 'complexity']].head())
print("Average Complexity : ", average_complexity)
print("Average Jaccard : ", average_jaccard)
print("Average Faithfulness : ", average_faithfulness)

print("\nMoyenne des valeurs ROAD :")
print(average_road)

# Tracé des moyennes ROAD
plt.figure(figsize=(10, 5))
x = list(average_road.keys())
y = list(average_road.values())

plt.plot(x, y, marker='o')
plt.xticks(rotation=45)
plt.xlabel('Perturbed %')
plt.ylabel('Average importance')
plt.title('Average ROAD feature importance')
plt.grid(True)
plt.tight_layout()
plt.show()
