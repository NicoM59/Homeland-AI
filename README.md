```
# 🧠 Homeland AI : Intelligence NLP Appliquée à la Santé Mentale

> Un dashboard clinique intelligent combinant Machine Learning robuste (Nested CV) et Large Language Models (RAG/OpenRouter).

---

## 📋 Présentation du Projet
Ce projet vise à fournir un outil d'aide au dépistage précoce des troubles mentaux à partir de données textuelles (posts, narrations patient). L'objectif est de transformer des données non structurées en **insights cliniques actionnables** tout en garantissant une rigueur statistique maximale.

### 🎯 Problématique Clinique
Comment assurer un tri (triage) fiable et explicable des pathologies mentales (ADHD, Bipolarité, Schizophrénie, etc.) malgré le déséquilibre des classes et la subtilité sémantique des témoignages ?

---

## 🛠️ Architecture Technique & "Muscles"
Le projet repose sur une approche hybride combinant performance classique et puissance générative.

### 1. Pipeline de Machine Learning (Baseline Robuste)
* **Modèle Champion** : LinearSVC avec pondération équilibrée des classes.
* **Validation Statistique** : Mise en œuvre d'une **Nested Cross-Validation** (K-Fold externe pour l'évaluation, GridSearch interne) pour éliminer tout biais de sélection et garantir un score généralisable.
* **Métrique Clé** : Focus sur le **Critical Recall** (94%+) pour minimiser les faux négatifs sur les pathologies à haut risque.

### 2. Module d'IA Générative (Copilot)
* **RAG (Retrieval-Augmented Generation)** : Système de questions-réponses basé sur la documentation technique du projet.
* **Client OpenRouter** : Intégration résiliente de LLM (GPT-4o mini) avec gestion des timeouts et retries pour l'explication des résultats cliniques.

---

## 🎨 Interface Premium (Dashboard)
L'application Streamlit utilise un design **Glassmorphism** moderne pour une expérience utilisateur de niveau "SaaS Santé" :
* **Overview** : Monitoring des métriques globales et état du dataset.
* **Inference** : Classification en temps réel avec barres de probabilités.
* **Monitoring** : Analyse des artefacts de validation (CSV) issus du pipeline d'entraînement.
* **Chat** : Assistant intelligent pour explorer la méthodologie du projet.

---

## 🚀 Installation & Déploiement

### Prérequis
* Python 3.10+
* Un environnement virtuel (ex: `ds_cpu`)

### Setup
1. **Cloner le projet**
   ```bash
   git clone [https://github.com/ton-profil/final_project_jedha.git](https://github.com/ton-profil/final_project_jedha.git)
   cd final_project_jedha
```

2. **Installer les dépendances**
   **Bash**

   ```
   pip install -r requirements_streamlit.txt
   ```
3. **Configuration (Secrets)**
   Créez un fichier `.env` à la racine :
   **Extrait de code**

   ```
   OPENROUTER_API_KEY=sk-or-v1-xxxxxx
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```
4. **Lancer l'application**
   **Bash**

   ```
   python -m streamlit run src/app/app.py
   ```

---

## ⚖️ Éthique & Limites

* **Non-Diagnostic** : Cet outil est un support à la décision et non un dispositif de diagnostic autonome.
* **Confidentialité** : Architecture conçue pour respecter la vie privée (gestion locale des modèles classiques).


## 👨‍💻 Auteurs

**Ana Gouveia et Nicolas Moignard - Promo DSFS-OD-14**
