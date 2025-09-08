#NOTE: this code was written with the help of Claude

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class PersonalDilemmaNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(PersonalDilemmaNN, self).__init__()
        
        #2-layer network
        hidden_size = 64
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

def train_neural_network(X_train, y_train, X_val, y_val, epochs=150, lr=0.001, run_num=0):
    """Train neural network with early stopping and cross-validation"""
    if run_num == 0:  # Only print detailed info for first run
        print("Training Neural Network with cross-validation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if run_num == 0:
        print(f"Using device: {device}")
    
    input_dim = X_train.shape[1]
    
    # Cross-validation for hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run_num)
    best_cv_score = 0
    best_params = {'dropout': 0.3, 'lr': 0.001}
    
    # Hyperparameter search
    param_combinations = [
        {'dropout': 0.2, 'lr': 0.001},
        {'dropout': 0.3, 'lr': 0.001},
        {'dropout': 0.4, 'lr': 0.001},
        {'dropout': 0.3, 'lr': 0.0005},
        {'dropout': 0.3, 'lr': 0.002}
    ]
    
    for params in param_combinations:
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            # Convert to tensors
            X_cv_train_tensor = torch.FloatTensor(X_cv_train).to(device)
            y_cv_train_tensor = torch.FloatTensor(y_cv_train).reshape(-1, 1).to(device)
            X_cv_val_tensor = torch.FloatTensor(X_cv_val).to(device)
            
            # Create model
            model = PersonalDilemmaNN(input_dim, params['dropout']).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
            
            # Train for fewer epochs in CV
            model.train()
            for epoch in range(50):  # Reduced epochs for CV
                optimizer.zero_grad()
                outputs = model(X_cv_train_tensor)
                loss = criterion(outputs, y_cv_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_cv_val_tensor)
                val_pred = (val_outputs > 0.5).float().cpu().numpy().flatten()
                cv_score = accuracy_score(y_cv_val, val_pred)
                cv_scores.append(cv_score)
        
        avg_cv_score = np.mean(cv_scores)
        if avg_cv_score > best_cv_score:
            best_cv_score = avg_cv_score
            best_params = params
    
    if run_num == 0:
        print(f"Best NN params: {best_params}, CV Score: {best_cv_score:.3f}")
    
    # Train final model with best parameters
    model = PersonalDilemmaNN(input_dim, best_params['dropout']).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        train_loss = criterion(outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_pred = (val_outputs > 0.5).float().cpu().numpy().flatten()
            val_acc = accuracy_score(y_val, val_pred)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if run_num == 0:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 30 == 0 and run_num == 0:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final validation accuracy
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_pred = (val_outputs > 0.5).float().cpu().numpy().flatten()
        final_val_acc = accuracy_score(y_val, val_pred)
    
    if run_num == 0:
        print(f"Final Neural Network validation accuracy: {final_val_acc:.3f}")
    
    return model, best_cv_score, device

def predict_with_nn(model, X, device):
    """Make predictions with trained neural network"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs > 0.5).float().cpu().numpy().flatten()
        probabilities = outputs.cpu().numpy().flatten()
    return predictions.astype(int), probabilities

def load_and_analyze_data(csv_path, run_num=0):
    if run_num == 0:
        print("Loading and analyzing CSV data...")
    df = pd.read_csv(csv_path)
    
    # Filter valid data
    valid_df = df.dropna(subset=['prosecution', 'defence', 'verdict'])
    valid_df = valid_df[valid_df['verdict'].isin(['guilty', 'innocent'])]
    
    if run_num == 0:
        print(f"Loaded {len(valid_df)} valid examples")
        
        # Analyze class distribution
        guilty_count = sum(valid_df['verdict'] == 'guilty')
        innocent_count = sum(valid_df['verdict'] == 'innocent')
        print(f"Class distribution: At Fault: {guilty_count} ({guilty_count/len(valid_df)*100:.1f}%), "
              f"Not At Fault: {innocent_count} ({innocent_count/len(valid_df)*100:.1f}%)")
        
        # Analyze text lengths
        pros_avg_len = valid_df['prosecution'].str.len().mean()
        def_avg_len = valid_df['defence'].str.len().mean()
        print(f"Average text lengths: Accusation: {pros_avg_len:.0f} chars, Defense: {def_avg_len:.0f} chars")
    
    if len(valid_df) < 50:
        raise ValueError(f"Not enough data for reliable prediction. Need at least 50 examples, got {len(valid_df)}")
    
    valid_df['label'] = (valid_df['verdict'] == 'guilty').astype(int)
    return valid_df

def extract_personal_dilemma_features(text):
    """Extract features relevant to personal moral dilemmas"""
    if pd.isna(text) or text == "":
        return np.zeros(15)
    
    text = str(text).lower()
    
    # Moral responsibility indicators
    responsibility_words = ['fault', 'blame', 'responsible', 'caused', 'mistake', 'error', 'wrong', 'should have', 'could have']
    
    # Emotional/relationship indicators
    emotional_words = ['hurt', 'upset', 'angry', 'sad', 'disappointed', 'betrayed', 'trust', 'love', 'hate', 'feel', 'emotion']
    
    # Justification/excuse indicators
    justification_words = ['because', 'reason', 'excuse', 'explain', 'understand', 'situation', 'circumstances', 'pressure', 'stress']
    
    # Intent/awareness indicators
    intent_words = ['meant to', 'intended', 'on purpose', 'accident', 'accidentally', 'deliberately', 'knew', 'aware', 'realize']
    
    # Apology/remorse indicators
    remorse_words = ['sorry', 'apologize', 'regret', 'wish', 'feel bad', 'ashamed', 'guilty', 'remorse']
    
    # Relationship context words
    relationship_words = ['friend', 'family', 'partner', 'relationship', 'together', 'close', 'trust', 'communicate']
    
    # Consequence/impact words
    consequence_words = ['consequence', 'result', 'impact', 'effect', 'damage', 'harm', 'help', 'benefit', 'outcome']
    
    features = [
        # Word category counts
        sum(1 for word in responsibility_words if word in text),
        sum(1 for word in emotional_words if word in text),
        sum(1 for word in justification_words if word in text),
        sum(1 for word in intent_words if word in text),
        sum(1 for word in remorse_words if word in text),
        sum(1 for word in relationship_words if word in text),
        sum(1 for word in consequence_words if word in text),
        
        # Text characteristics
        len(text),  # Length
        len(text.split()),  # Word count
        len([s for s in text.split('.') if s.strip()]),  # Sentence count
        text.count('!'),  # Exclamation marks (emotion)
        text.count('?'),  # Question marks (uncertainty/seeking understanding)
        
        # Personal pronouns (ownership/responsibility)
        text.count(' i ') + text.count('i '),  # First person
        text.count(' my ') + text.count(' me '),  # Personal ownership
        
        # Complexity measure
        np.mean([len(word) for word in text.split()]) if text.split() else 0
    ]
    
    return np.array(features)

def create_advanced_features(df, run_num=0):
    if run_num == 0:
        print("Creating advanced personal dilemma features...")
    
    # Separate accusation and defence analysis
    pros_features = np.array([extract_personal_dilemma_features(text) for text in df['prosecution']])
    def_features = np.array([extract_personal_dilemma_features(text) for text in df['defence']])
    
    # Create comparative features (accusation vs defence)
    comparative_features = pros_features - def_features
    ratio_features = np.divide(pros_features, def_features + 1, 
                              out=np.zeros_like(pros_features), 
                              where=(def_features + 1) != 0)
    
    # TF-IDF with personal dilemma focus
    # Accusation TF-IDF
    pros_vectorizer = TfidfVectorizer(
        max_features=80,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words 3+ chars
    )
    
    # Defence TF-IDF  
    def_vectorizer = TfidfVectorizer(
        max_features=80,
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    pros_tfidf = pros_vectorizer.fit_transform(df['prosecution'].fillna('')).toarray()
    def_tfidf = def_vectorizer.fit_transform(df['defence'].fillna('')).toarray()
    
    # Combine all features
    all_features = np.hstack([
        pros_tfidf,           # 80 features
        def_tfidf,            # 80 features  
        pros_features,        # 15 features
        def_features,         # 15 features
        comparative_features, # 15 features
        ratio_features        # 15 features
    ])
    
    if run_num == 0:
        print(f"Created {pros_tfidf.shape[1]} accusation TF-IDF features")
        print(f"Created {def_tfidf.shape[1]} defence TF-IDF features")
        print(f"Created {pros_features.shape[1]} accusation semantic features")
        print(f"Created {def_features.shape[1]} defence semantic features")
        print(f"Created {comparative_features.shape[1]} comparative features")
        print(f"Created {ratio_features.shape[1]} ratio features")
        print(f"Total raw features: {all_features.shape[1]}")
    
    # Feature selection using mutual information
    if len(df) > 50:
        selector = SelectKBest(mutual_info_classif, k=min(140, all_features.shape[1]))
        selected_features = selector.fit_transform(all_features, df['label'])
        if run_num == 0:
            print(f"Selected {selected_features.shape[1]} most informative features")
        return selected_features, (pros_vectorizer, def_vectorizer), selector
    
    return all_features, (pros_vectorizer, def_vectorizer), None

def train_optimized_ensemble(X_train, y_train, X_val, y_val, run_num=0):
    if run_num == 0:
        print("\nTraining optimized ensemble with hyperparameter tuning...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {}
    cv_scores = {}
    
    # Cross-validation setup with different random state per run
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run_num)
    
    # 1. Hyperparameter tuning for Logistic Regression
    if run_num == 0:
        print("Training Logistic Regression with cross-validation...")
    best_lr_score = 0
    best_lr_C = 1.0
    
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        lr = LogisticRegression(C=C, random_state=42 + run_num, max_iter=1000, class_weight='balanced')
        scores = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        avg_score = scores.mean()
        if avg_score > best_lr_score:
            best_lr_score = avg_score
            best_lr_C = C
    
    final_lr = LogisticRegression(C=best_lr_C, random_state=42 + run_num, max_iter=1000, class_weight='balanced')
    final_lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = final_lr
    cv_scores['Logistic Regression'] = best_lr_score
    
    if run_num == 0:
        print(f"  Best C: {best_lr_C}, CV Score: {best_lr_score:.3f}")
    
    # 2. Hyperparameter tuning for Random Forest
    if run_num == 0:
        print("Training Random Forest with cross-validation...")
    best_rf_score = 0
    best_rf_params = {}
    
    param_combinations = [
        {'n_estimators': 30, 'max_depth': 3, 'min_samples_split': 8},
        {'n_estimators': 50, 'max_depth': 4, 'min_samples_split': 6},
        {'n_estimators': 30, 'max_depth': 5, 'min_samples_split': 8},
        {'n_estimators': 20, 'max_depth': 3, 'min_samples_split': 10}
    ]
    
    for params in param_combinations:
        rf = RandomForestClassifier(
            **params,
            min_samples_leaf=3,
            random_state=42 + run_num, 
            class_weight='balanced'
        )
        scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
        avg_score = scores.mean()
        if avg_score > best_rf_score:
            best_rf_score = avg_score
            best_rf_params = params
    
    final_rf = RandomForestClassifier(
        **best_rf_params,
        min_samples_leaf=3,
        random_state=42 + run_num, 
        class_weight='balanced'
    )
    final_rf.fit(X_train, y_train)
    models['Random Forest'] = final_rf
    cv_scores['Random Forest'] = best_rf_score
    
    if run_num == 0:
        print(f"  Best params: {best_rf_params}, CV Score: {best_rf_score:.3f}")
    
    # 3. Train Neural Network
    nn_model, nn_cv_score, device = train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, run_num=run_num)
    models['Neural Network'] = (nn_model, device)  # Store model and device
    cv_scores['Neural Network'] = nn_cv_score
    
    return models, cv_scores, scaler

def evaluate_models_for_plotting(models, cv_scores, scaler, X_test, y_test, run_num=0):
    """Evaluate models and return test accuracies for plotting"""
    X_test_scaled = scaler.transform(X_test)
    
    test_scores = {}
    
    # Evaluate each model
    for name, model in models.items():
        if name == 'Logistic Regression':
            pred = model.predict(X_test_scaled)
        elif name == 'Random Forest':
            pred = model.predict(X_test)
        elif name == 'Neural Network':
            nn_model, device = model  # Unpack model and device
            pred, _ = predict_with_nn(nn_model, X_test_scaled, device)
        
        test_scores[name] = accuracy_score(y_test, pred)
    
    if run_num == 0:
        print(f"\nRun {run_num + 1} Test Accuracies:")
        for name, score in test_scores.items():
            print(f"  {name}: {score:.3f}")
    
    return test_scores

def plot_results(results_df):
    """Create 3 bar plots showing test accuracies across 10 runs for each model"""
    # Create 3 separate bar plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Test Accuracy Across 10 Runs for Each Model', fontsize=16, fontweight='bold')
    
    runs = list(range(1, 11))
    
    # Bar plot for Logistic Regression
    ax1.bar(runs, results_df['Logistic Regression'], color='#2E86C1', alpha=0.8, edgecolor='black')
    ax1.set_title('Logistic Regression', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(runs)
    
    # Add value labels on bars
    for i, v in enumerate(results_df['Logistic Regression']):
        ax1.text(i+1, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=5)
    
    # Bar plot for Random Forest
    ax2.bar(runs, results_df['Random Forest'], color='#E74C3C', alpha=0.8, edgecolor='black')
    ax2.set_title('Random Forest', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(runs)
    
    # Add value labels on bars
    for i, v in enumerate(results_df['Random Forest']):
        ax2.text(i+1, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=5)
    
    # Bar plot for Neural Network
    ax3.bar(runs, results_df['Neural Network'], color='#28B463', alpha=0.8, edgecolor='black')
    ax3.set_title('Neural Network', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Run Number')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(runs)
    
    # Add value labels on bars
    for i, v in enumerate(results_df['Neural Network']):
        ax3.text(i+1, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=5)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    methods = ['Logistic Regression', 'Random Forest', 'Neural Network']
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS ACROSS 10 RUNS")
    print("="*80)
    
    for method in methods:
        data = results_df[method]
        print(f"\n{method}:")
        print(f"  Mean Accuracy: {data.mean():.3f} (±{data.std():.3f})")
        print(f"  Range: {data.min():.3f} - {data.max():.3f}")
        print(f"  Median: {data.median():.3f}")
        print(f"  Consistency: {'High' if data.std() < 0.05 else 'Moderate' if data.std() < 0.1 else 'Low'}")
    
    # Model ranking
    print(f"\nModel Ranking (by mean accuracy):")
    method_means = [(method, results_df[method].mean()) for method in methods]
    method_means.sort(key=lambda x: x[1], reverse=True)
    
    for i, (method, mean_acc) in enumerate(method_means, 1):
        print(f"  {i}. {method}: {mean_acc:.3f}")
    
    # Best overall run
    print(f"\nBest Overall Run:")
    best_run_idx = -1
    best_avg = -1
    for run in range(10):
        run_avg = np.mean([results_df[method].iloc[run] for method in methods])
        if run_avg > best_avg:
            best_avg = run_avg
            best_run_idx = run
    
    print(f"  Run {best_run_idx + 1}: Average accuracy = {best_avg:.3f}")
    for method in methods:
        print(f"    {method}: {results_df[method].iloc[best_run_idx]:.3f}")

def run_single_experiment(csv_path, run_num):
    """Run a single experiment and return test accuracies"""
    df = load_and_analyze_data(csv_path, run_num)
    features, vectorizers, selector = create_advanced_features(df, run_num)
    
    # Split data with different random state for each run
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, df['label'].values,
        test_size=0.25,
        random_state=42 + run_num,  # Different random state for each run
        stratify=df['label'].values
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42 + run_num,  # Different random state for each run
        stratify=y_temp
    )
    
    if run_num == 0:
        print(f"\nData split:")
        print(f"Training: {len(X_train)} examples ({len(X_train)/len(features)*100:.1f}%)")
        print(f"Validation: {len(X_val)} examples ({len(X_val)/len(features)*100:.1f}%)")
        print(f"Test: {len(X_test)} examples ({len(X_test)/len(features)*100:.1f}%)")
    
    models, cv_scores, scaler = train_optimized_ensemble(X_train, y_train, X_val, y_val, run_num)
    
    if len(X_test) > 0:
        test_scores = evaluate_models_for_plotting(models, cv_scores, scaler, X_test, y_test, run_num)
        return test_scores
    else:
        print(f"Run {run_num + 1}: Insufficient test data")
        return None

def main():
    csv_path = "guardian_dataset.csv" 
    num_runs = 10
    
    print("PERSONAL DILEMMA VERDICT PREDICTOR - MULTIPLE RUNS ANALYSIS")
    print("=" * 70)
    print(f"Running {num_runs} experiments with different random seeds...")
    print("=" * 70)
    
    all_results = {
        'Logistic Regression': [],
        'Random Forest': [],
        'Neural Network': []
    }
    
    try:
        for run in range(num_runs):
            print(f"\n{'='*50}")
            print(f"RUNNING EXPERIMENT {run + 1}/{num_runs}")
            print(f"{'='*50}")
            
            test_scores = run_single_experiment(csv_path, run)
            
            if test_scores is not None:
                all_results['Logistic Regression'].append(test_scores['Logistic Regression'])
                all_results['Random Forest'].append(test_scores['Random Forest'])
                all_results['Neural Network'].append(test_scores['Neural Network'])
                
                # Print brief summary for this run
                print(f"\nRun {run + 1} Results:")
                for method, accuracy in test_scores.items():
                    print(f"  {method}: {accuracy:.3f}")
            
            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # DataFrame
        results_df = pd.DataFrame(all_results)
        
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETED - GENERATING ANALYSIS")
        print(f"{'='*70}")
        plot_results(results_df)
            
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_path}'. Please check the file path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()