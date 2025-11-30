import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gen_data.gen_expert_ensemble import HybridEnsembleExpert
from utils.risk_profile import build_risk_profile

def test_stock_allocation_sensitivity():
    print("--- 🧠 Testing Expert Stock Allocation Sensitivity ---")
    
    # 1. Create a "Clear Winner" Market Scenario
    # 20 Stocks total.
    # Stocks 0-4:  High Return (2%), Low Volatility (1%)  -> "The Winners"
    # Stocks 5-19: Zero Return (0%), High Volatility (3%) -> "The Losers"
    n_stocks = 20
    np.random.seed(42)
    
    returns = np.concatenate([
            np.random.normal(0.02, 0.01, 5),  # Winners (Sharpe ~2.0)
            np.random.normal(0.005, 0.003, 15)   # Losers (Sharpe ~0.0)
    ])
    
    # Correlation: Winners correlate with Winners.
    corr = np.eye(n_stocks)
    corr[:5, :5] = 0.8
    np.fill_diagonal(corr, 1.0)
    
    ind = np.zeros((n_stocks, n_stocks))
    
    # 2. Run Experts with different Risk Scores
    scenarios = [
        {'score': 0.1, 'label': 'Conservative'},
        {'score': 0.9, 'label': 'Aggressive'}
    ]
    
    results = {}
    
    for s in scenarios:
        print(f"\n🔎 Generating Portfolio for {s['label']} (Risk={s['score']})...")
        
        profile = build_risk_profile(s['score'])
        # Disable randomization to test pure logic
        expert = HybridEnsembleExpert(risk_profile=profile, randomize_params=False)
        
        weights = expert.generate_expert_action(
            returns=returns,
            correlation_matrix=corr,
            industry_matrix=ind
        )
        
        results[s['label']] = weights
        
        # Calculate Concentration (HHI)
        # HHI = Sum(Weight^2). Higher = More Concentrated.
        hhi = np.sum(weights**2)
        
        # Calculate Allocation to Winners (Stocks 0-4)
        winner_alloc = np.sum(weights[:5])
        
        print(f"   -> Allocation to Winners (Top 5): {winner_alloc:.1%}")
        print(f"   -> Concentration Score (HHI): {hhi:.4f}")
        
    # 3. VERIFICATION (The Pass/Fail Criteria)
    print("\n--- 📝 Logic Verdict ---")
    
    agg_w = results['Aggressive']
    con_w = results['Conservative']
    
    # Check 1: Aggressive should have HIGHER concentration (HHI)
    hhi_agg = np.sum(agg_w**2)
    hhi_con = np.sum(con_w**2)
    
    # Check 2: Aggressive should allow higher max weights (Peaks)
    max_agg = np.max(agg_w)
    max_con = np.max(con_w)
    
    print(f"Aggressive HHI: {hhi_agg:.4f} | Max Weight: {max_agg:.1%}")
    print(f"Conservative HHI: {hhi_con:.4f} | Max Weight: {max_con:.1%}")
    
    if hhi_agg > hhi_con * 1.2:
        print("✅ PASS: Aggressive Expert is significantly more concentrated.")
    else:
        print("❌ FAIL: Experts look identical (Dynamic weighting failed).")
        
    if max_agg > max_con:
        print("✅ PASS: Aggressive Expert takes bigger bets (Higher Max Weight).")
    else:
        print("❌ FAIL: Constraints are not adapting.")

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    indices = np.arange(n_stocks)
    width = 0.35
    
    plt.bar(indices - width/2, agg_w, width, label='Aggressive (Risk 0.9)', color='red', alpha=0.7)
    plt.bar(indices + width/2, con_w, width, label='Conservative (Risk 0.1)', color='blue', alpha=0.7)
    
    # Mark the Winners
    plt.axvspan(-0.5, 4.5, color='green', alpha=0.1, label='Winner Stocks (0-4)')
    
    plt.title("Expert Stock Allocation: Aggressive vs Conservative")
    plt.xlabel("Stock Index")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("test/expert_stock_sensitivity.png")
    print("\nChart saved to: test/expert_stock_sensitivity.png")

if __name__ == "__main__":
    test_stock_allocation_sensitivity()