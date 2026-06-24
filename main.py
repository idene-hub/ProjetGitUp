import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000/predire"

# je souhaite  Définir la véritable fonction principale de l'application Streamlit pour tester mon projet
def main():
    st.title("Prédiction des Maladies Cardiaques")
    st.write("Cette application prédit la probabilité d'une maladie cardiaque basée sur vos entrées.")

    # Créer des onglets pour la prédiction individuelle et la prédiction par fichier
    onglet1, onglet2 = st.tabs(["Prédiction Individuelle", "Prédiction par Fichier"])

    with onglet1:
        # Formulaire pour saisir les valeurs des attributs
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sexe", options=[0, 1], format_func=lambda x: "Homme" if x == 1 else "Femme")
        cp = st.selectbox("Type de Douleur Thoracique (cp)", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
        trestbps = st.number_input("Pression Artérielle au Repos (trestbps)", 80, 200, 120)
        chol = st.number_input("Cholestérol Sérique (chol)", 100, 600, 200)
        fbs = st.selectbox("Glycémie à Jeun > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
        restecg = st.selectbox("Résultats de l'Électrocardiogramme au Repos (restecg)", options=[0, 1, 2])
        thalach = st.number_input("Fréquence Cardiaque Maximale (thalach)", 70, 250, 150)
        exang = st.selectbox("Angine Induite par l'Exercice (exang)", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
        oldpeak = st.number_input("Dépression ST induite par l'exercice (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Pente du Segment ST (slope)", options=[0, 1, 2])
        ca = st.selectbox("Nombre de Vaisseaux Colorés (ca)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", options=[1, 2, 3], format_func=lambda x: f"Type {x}")

        # Créer un dictionnaire avec les données d'entrée
        donnees_entree = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        # Bouton pour faire la prédiction
        if st.button("Prédire", key="predict_individual"):
            try:
                # Envoyer une requête POST à l'API FastAPI
                reponse = requests.post(API_URL, json=donnees_entree)
                
                # Vérifier si la requête a réussi
                if reponse.status_code == 200:
                    resultat = reponse.json()
                    st.success(f"Probabilité de maladie cardiaque : {resultat['probabilite_maladie_cardiaque']:.2f}")
                    st.write(f"Diagnostic : {resultat['diagnostic']}")
                else:
                    st.error("Erreur lors de la prédiction. Veuillez vérifier l'API.")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API : {e}")

    with onglet2:
        st.write("Importez un fichier (TXT, CSV, JSON, Excel) pour faire des prédictions sur plusieurs lignes.")
        fichier = st.file_uploader("Choisissez un fichier", type=["txt", "csv", "json", "xlsx"])

        if fichier is not None:
            try:
                # Déterminer le type de fichier et charger les données
                if fichier.name.endswith(".csv"):
                    donnees = pd.read_csv(fichier)
                elif fichier.name.endswith(".json"):
                    donnees = pd.read_json(fichier)
                elif fichier.name.endswith(".xlsx"):
                    donnees = pd.read_excel(fichier)
                elif fichier.name.endswith(".txt"):
                    donnees = pd.read_csv(fichier, delimiter="\t")
                else:
                    st.error("Format de fichier non pris en charge.")
                    return

                st.write("Aperçu des données :", donnees.head())

                # Bouton pour lancer la prédiction sur le fichier
                if st.button("Prédire sur toutes les lignes", key="predict_file"):
                    # Faire les prédictions sur toutes les lignes
                    predictions = []
                    probabilites = []
                    for _, ligne in donnees.iterrows():
                        donnees_entree = ligne.to_dict()
                        reponse = requests.post(API_URL, json=donnees_entree)
                        if reponse.status_code == 200:
                            resultat = reponse.json()
                            predictions.append(resultat["diagnostic"])
                            probabilites.append(resultat["probabilite_maladie_cardiaque"])
                        else:
                            predictions.append("Erreur")
                            probabilites.append(None)

                    # Ajouter les prédictions et les probabilités au DataFrame
                    donnees["Prediction"] = predictions
                    donnees["Probabilite_Maladie_Cardiaque"] = probabilites
                    st.write("Données avec Prédictions :", donnees)

                    # Télécharger le fichier avec les prédictions
                    csv = donnees.to_csv(index=False).encode('utf-8')
                    b64 = base64.b64encode(csv).decode()  # some strings <-> bytes conversions necessary here
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Télécharger les prédictions</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # Calculer la répartition des diagnostics
                    repartition = donnees["Prediction"].value_counts()
                    proportions = (repartition / len(donnees)) * 100
                    st.write("Répartition des prédictions :", repartition)

                    # Afficher un graphique de répartition avec les proportions
                    fig, ax = plt.subplots()
                    bars = ax.bar(repartition.index, repartition.values)
                    for bar, proportion in zip(bars, proportions):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{proportion:.1f}%', ha='center', va='center', color='white', fontsize=10)
                    ax.set_title("Répartition des Prédictions avec Proportions")
                    ax.set_xlabel("Diagnostic")
                    ax.set_ylabel("Nombre de Cas")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier : {e}")

# Exécuter l'application Streamlit
if __name__ == "__main__":
    main()
