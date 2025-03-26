from flask import Flask, render_template, request, redirect, session, url_for, flash
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
import numpy as np
from pymongo import MongoClient
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ✅ MySQL (AWS RDS)
db = mysql.connector.connect(
    host="database-1.cvakiyk6o3wb.us-east-2.rds.amazonaws.com",
    user="admin",
    password="ChemicalCompound4!",
    database="userInfo"
)
cursor = db.cursor()

# ✅ MongoDB Atlas connection
mongo_uri = "mongodb+srv://myuser:ChemicalCompound4!@chemicalcompoundhistory.6hdzk.mongodb.net/?retryWrites=true&w=majority&appName=ChemicalCompoundHistory"
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client["ChemicalCompoundHistory"]
history_collection = mongo_db["search_history"]

# ✅ Load models
alogp_model = joblib.load('model/alogp_rf_model.pkl')
cxlogp_model = joblib.load('model/cx_logp_rf_model.pkl')
mw_model = joblib.load('model/molecular_weight_rf_model.pkl')
psa_model = joblib.load('model/polar_surface_area_rf_model.pkl')
rot_model = joblib.load('model/rotatable_bonds_rf_model.pkl')

# ✅ Feature extraction
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.HeavyAtomCount(mol),
        0
    ]

@app.route('/')
def index():
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            flash('Email already registered')
            return redirect('/register')

        cursor.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password))
        db.commit()
        flash('Registered successfully. Please log in.')
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            session['user'] = email
            return redirect('/home')
        else:
            flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'user' not in session:
        return redirect('/login')

    predictions = None
    error = None

    if request.method == 'POST':
        smiles = request.form['smiles']
        features = extract_features(smiles)

        if features:
            features = np.array(features).reshape(1, -1)
            predictions = {
                'ALogP': float(alogp_model.predict(features)[0]),
                'CX_LogP': float(cxlogp_model.predict(features)[0]),
                'Molecular Weight': float(mw_model.predict(features)[0]),
                'Polar Surface Area': float(psa_model.predict(features)[0]),
                'Rotatable Bonds': float(rot_model.predict(features)[0]),
            }

            # ✅ Store in MongoDB with standard types
            history_collection.insert_one({
                "email": session['user'],
                "smiles": smiles,
                "predictions": predictions
            })
        else:
            error = "Invalid SMILES notation."

    return render_template('home.html', user=session['user'], predictions=predictions, error=error)

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    user_email = session['user']
    results = list(history_collection.find({"email": user_email}))
    return render_template('history.html', user=user_email, records=results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)





