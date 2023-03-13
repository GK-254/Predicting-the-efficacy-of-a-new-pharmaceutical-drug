# Predicting-the-efficacy-of-a-new-pharmaceutical-drug
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from flask import Flask, request, jsonify

# Load dataset
df = pd.read_csv('pharmaceuticals.csv')

# Preprocessing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['Efficacy'] = df['Efficacy'].map({'High': 1, 'Low': 0})

# Feature engineering
df['Mol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
df['MolWt'] = df['Mol'].apply(lambda x: Descriptors.MolWt(x))
df['NumAtoms'] = df['Mol'].apply(lambda x: x.GetNumAtoms())
df['LogP'] = df['Mol'].apply(lambda x: Descriptors.MolLogP(x))

# Model development
X = df[['MolWt', 'NumAtoms', 'LogP']]
y = df['Efficacy']
rf = RandomForestClassifier()
scores = cross_val_score(rf, X, y, cv=5)
rf.fit(X, y)
y_pred = rf.predict(X)
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Model deployment
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    mol = Chem.MolFromSmiles(data['smiles'])
    molwt = Descriptors.MolWt(mol)
    numatoms = mol.GetNumAtoms()
    logp = Descriptors.MolLogP(mol)
    pred = rf.predict([[molwt, numatoms, logp]])
    return jsonify({'prediction': int(pred[0])})

if __name__ == '__main__':
    app.run()
