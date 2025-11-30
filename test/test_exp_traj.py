import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gen_data.gen_expert_ensemble import HybridEnsembleExpert
from utils.risk_profile import build_risk_profile

def test_expert_logic():
    print("--- 🧠 Testing Expert Strategy Internals (White Box) ---")
    
    # Define Risk Profiles to test
    scenarios = [
        {'score': 0.1, 'label': 'Conservative'},
        {'score': 0.5, 'label': 'Moderate'},
        {'score': 0.9, 'label': 'Aggressive'}
    ]
    
    # Dummy Market Data (50 Stocks)
    n_stocks = 50
    np.random.seed(42)
    returns = np.random.randn(n_stocks) * 0.02 + 0.005 # Slight positive drift
    # Make stock 0 a clear winner to tempt the aggressive expert
    returns[0] = 0.10 
    
    # Covariance / Correlation
    corr = np.eye(n_stocks) * 0.5 + 0.5 # High correlation
    ind = np.eye(n_stocks) # Dummy industry
    
    internal_data = []
    
    for s in scenarios:
        print(f"\n🔎 Inspecting {s['label']} Expert (Risk={s['score']})...")
        
        # 1. Initialize Expert
        profile = build_risk_profile(s['score'])
        expert = HybridEnsembleExpert(risk_profile=profile, randomize_params=False)
        
        # 2. CHECK INTERNALS (The Proof)
        # We look at the private attribute 'ensemble_weights'
        strat_weights = expert.ensemble_weights
        print(f"   Strategy Mix: {strat_weights}")
        
        # 3. Generate Action
        w = expert.generate_expert_action(returns, corr, ind)
        
        # 4. Calculate Concentration (HHI)
        # HHI = Sum(Weight^2). 1.0 = All-in, 0.0 = Equal weight
        hhi = np.sum(w**2)
        print(f"   Portfolio Concentration (HHI): {hhi:.4f}")
        
        internal_data.append({
            'Label': s['label'],
            'Markowitz_Trust': strat_weights.get('robust_markowitz', 0),
            'RiskParity_Trust': strat_weights.get('risk_parity', 0),
            'HHI': hhi
        })

    # --- VERIFICATION ---
    print("\n--- 📝 Logic Verdict ---")
    
    # Check 1: Does Aggressive trust Markowitz more?
    agg_trust = internal_data[2]['Markowitz_Trust']
    con_trust = internal_data[0]['Markowitz_Trust']
    
    if agg_trust > con_trust * 2:
        print("✅ PASS: Aggressive Expert heavily favors 'Markowitz' (Profit Seeking).")
    else:
        print(f"❌ FAIL: Strategies look similar (Agg: {agg_trust:.2f}, Con: {con_trust:.2f})")

    # Check 2: Does Conservative trust Risk Parity more?
    agg_rp = internal_data[2]['RiskParity_Trust']
    con_rp = internal_data[0]['RiskParity_Trust']
    
    if con_rp > agg_rp * 2:
        print("✅ PASS: Conservative Expert heavily favors 'Risk Parity' (Safety).")
    else:
        print(f"❌ FAIL: Risk Parity weights are similar.")

    # --- VISUALIZATION ---
    labels = [d['Label'] for d in internal_data]
    markowitz = [d['Markowitz_Trust'] for d in internal_data]
    risk_parity = [d['RiskParity_Trust'] for d in internal_data]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, markowitz, width, label='Markowitz (Profit)', color='red', alpha=0.7)
    plt.bar(x + width/2, risk_parity, width, label='Risk Parity (Safety)', color='blue', alpha=0.7)
    
    plt.title("Expert Brain Scan: Str   ategy Mixing by Risk Profile")
    plt.ylabel("Internal Strategy Weight")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("test/expert_internal_logic.png")
    print("\nChart saved to: test/expert_internal_logic.png")

if __name__ == "__main__":
    test_expert_logic()