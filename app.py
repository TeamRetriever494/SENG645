from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
import numpy as np
from pymongo import MongoClient
import re
import os
from flask import send_from_directory

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MySQL (AWS RDS) connection
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

# Load ML models
alogp_model = joblib.load('model/alogp_rf_model.pkl')
cxlogp_model = joblib.load('model/cx_logp_rf_model.pkl')
mw_model = joblib.load('model/molecular_weight_rf_model.pkl')
psa_model = joblib.load('model/polar_surface_area_rf_model.pkl')
rot_model = joblib.load('model/rotatable_bonds_rf_model.pkl')

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
        password = request.form['password']

        # Password strength check
        if not re.search(r'[A-Z]', password) or not re.search(r'\d', password):
            flash('Password must contain at least one uppercase letter and one number.')
            return redirect('/register')

        password_hash = generate_password_hash(password)

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            flash('Email already registered')
            return redirect('/register')

        cursor.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password_hash))
        db.commit()
        flash('Registered successfully, please log in')
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[2], password):  # user[2] is password_hash
            session['user'] = email
            return redirect('/home')
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'user' not in session:
        return redirect('/login')

    predictions = None
    smiles = None
    mol_block = ""

    if request.method == 'POST':
        smiles = request.form['smiles']
        features = extract_features(smiles)

        if features:
            features = np.array(features).reshape(1, -1)
            predictions = {
                'ALogP': float(alogp_model.predict(features)[0]),
                'CxLogP': float(cxlogp_model.predict(features)[0]),
                'Molecular Weight': float(mw_model.predict(features)[0]),
                'Polar Surface Area': float(psa_model.predict(features)[0]),
                'Rotatable Bonds': float(rot_model.predict(features)[0]),
            }

            # Save to MongoDB
            history_collection.insert_one({
                "user": session['user'],
                "smiles": smiles,
                "predictions": predictions
            })

            # Generate MolBlock for 3D rendering
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol_block = Chem.MolToMolBlock(mol)

        else:
            flash("Invalid SMILES notation.")

    return render_template("home.html", user=session['user'], predictions=predictions, smiles=smiles, mol_block=mol_block)

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    cursor = history_collection.find({"user": session['user']})
    records = list(cursor)  # Fix: convert MongoDB cursor to list
    return render_template("history.html", records=records)

@app.route('/delete_history/<record_id>', methods=['POST'])
def delete_history(record_id):
    from bson import ObjectId
    history_collection.delete_one({"_id": ObjectId(record_id)})
    return redirect('/history')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment, default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)

# Serve manifest.json from the static folder
@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json', mimetype='application/json')

# Serve service-worker.js from the static folder
@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('static', 'service-worker.js', mimetype='application/javascript')





