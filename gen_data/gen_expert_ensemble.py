"""
Hybrid Ensemble Expert Strategy - Risk-Dynamic Allocation & Parameter Scaling
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.covariance import LedoitWolf, OAS
import warnings
warnings.filterwarnings('ignore')


class RobustMarkowitz:
    """Robust Markowitz with regularization and shrinkage"""
    
    def __init__(self, risk_aversion=1.0, regularization=0.01, shrinkage_method='ledoit_wolf'):
        self.risk_aversion = risk_aversion
        self.regularization = regularization
        self.shrinkage_method = shrinkage_method
    
    def estimate_covariance(self, returns):
        """Estimate covariance matrix with shrinkage"""
        if self.shrinkage_method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
        elif self.shrinkage_method == 'oas':
            oas = OAS()
            cov_matrix = oas.fit(returns).covariance_
        else:
            cov_matrix = np.cov(returns.T)
        
        # Add regularization
        cov_matrix += self.regularization * np.eye(len(cov_matrix))
        return cov_matrix
    
    def optimize(self, returns, expected_returns=None, constraints=None):
        """Optimize portfolio weights"""
        n_assets = returns.shape[1] if len(returns.shape) > 1 else len(returns)
        
        # Use historical mean if expected returns not provided
        if expected_returns is None:
            if len(returns.shape) > 1:
                expected_returns = returns.mean(axis=0)
            else:
                expected_returns = returns
        
        # Ensure returns is 2D for covariance estimation
        if len(returns.shape) == 1:
            returns_2d = np.tile(returns, (5, 1))
            returns_2d += np.random.randn(*returns_2d.shape) * 0.01
        else:
            returns_2d = returns
        
        cov_matrix = self.estimate_covariance(returns_2d)
        
        # Constraints
        if constraints is None:
            constraints = {}
        max_weight = constraints.get('max_weight', 0.3)
        min_weight = constraints.get('min_weight', 0.01)
        
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            # Maximize: Return - lambda * Variance
            return -portfolio_return + self.risk_aversion * portfolio_variance
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        weights = np.array(result.x, dtype=float)
        
        # Cleanup
        weights[weights < min_weight] = 0.0
        weights /= (weights.sum() + 1e-8)
        
        return weights


class BlackLitterman:
    """Black-Litterman model for incorporating momentum views"""
    
    def __init__(self, tau=0.05, risk_aversion=2.5):
        self.tau = tau
        self.risk_aversion = risk_aversion
    
    def generate_views(self, returns, correlation_matrix, industry_matrix, n_views=5):
        """Generate momentum-based views"""
        n_assets = len(returns)
        top_performers = np.argsort(-returns)[:n_views]
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, idx in enumerate(top_performers):
            P[i, idx] = 1.0
            Q[i] = returns[idx] * 1.2 # Expect momentum to continue
        
        return P, Q
    
    def optimize(self, returns, cov_matrix, correlation_matrix, industry_matrix, 
                 market_cap_weights=None, constraints=None):
        n_assets = len(returns)
        if market_cap_weights is None:
            market_cap_weights = np.ones(n_assets) / n_assets
        
        eps = 1e-4
        cov_matrix = np.array(cov_matrix, dtype=float) + eps * np.eye(n_assets)

        pi = self.risk_aversion * np.dot(cov_matrix, market_cap_weights)
        P, Q = self.generate_views(returns, correlation_matrix, industry_matrix)
        
        omega = np.dot(np.dot(P, self.tau * cov_matrix), P.T)
        omega = omega + eps * np.eye(omega.shape[0])
        
        tau_cov = self.tau * cov_matrix
        inv_tau_cov = np.linalg.pinv(tau_cov)
        inv_omega = np.linalg.pinv(omega)
        
        M_inverse = inv_tau_cov + np.dot(np.dot(P.T, inv_omega), P)
        M = np.linalg.pinv(M_inverse)
        mu_bl = np.dot(M, np.dot(inv_tau_cov, pi) + np.dot(np.dot(P.T, inv_omega), Q))
        
        markowitz = RobustMarkowitz(risk_aversion=self.risk_aversion)
        weights = markowitz.optimize(returns, expected_returns=mu_bl, constraints=constraints)
        
        return weights


class RiskParity:
    """Risk Parity / Equal Risk Contribution"""
    
    def optimize(self, returns, constraints=None):
        if len(returns.shape) == 1:
            returns_2d = np.tile(returns, (5, 1)) + np.random.randn(5, len(returns)) * 0.01
        else:
            returns_2d = returns
        
        cov_matrix = np.cov(returns_2d.T)
        n_assets = cov_matrix.shape[0]
        
        if constraints is None: constraints = {}
        max_weight = constraints.get('max_weight', 0.3)
        min_weight = constraints.get('min_weight', 0.01)
        
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / (portfolio_vol + 1e-8)
            target_rc = np.ones(n_assets) / n_assets
            return np.sum((risk_contrib - target_rc) ** 2)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0.0, max_weight) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        weights = np.array(result.x, dtype=float)
        
        weights[weights < min_weight] = 0.0
        weights /= (weights.sum() + 1e-8)
        
        return weights


class HRP:
    """Hierarchical Risk Parity"""
    
    def get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = pd.Index(range(0, sort_ix.shape[0] * 2, 2))
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]).sort_index()
            sort_ix.index = pd.Index(range(sort_ix.shape[0]))
        return sort_ix.tolist()
    
    def get_cluster_var(self, cov, cluster_items):
        cov_slice = cov.iloc[cluster_items, cluster_items]
        diag = np.diag(cov_slice)
        w = 1.0 / diag
        w /= w.sum()
        return np.dot(w, np.dot(cov_slice, w))
    
    def get_rec_bipart(self, cov, sort_ix):
        w = pd.Series(1.0, index=sort_ix)
        cluster_items = [sort_ix]
        while len(cluster_items) > 0:
            cluster_items = [i[j:k] for i in cluster_items 
                           for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
                           if len(i) > 1]
            for i in range(0, len(cluster_items), 2):
                cluster0 = cluster_items[i]
                cluster1 = cluster_items[i + 1]
                var0 = self.get_cluster_var(cov, cluster0)
                var1 = self.get_cluster_var(cov, cluster1)
                alpha = 1 - var0 / (var0 + var1)
                w[cluster0] *= alpha
                w[cluster1] *= 1 - alpha
        return w
    
    def optimize(self, returns, constraints=None):
        if len(returns.shape) == 1:
            returns_2d = np.tile(returns, (5, 1)) + np.random.randn(5, len(returns)) * 0.01
        else:
            returns_2d = returns
        
        cov_matrix = np.cov(returns_2d.T)
        corr_matrix = np.corrcoef(returns_2d.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        if constraints is None: constraints = {}
        min_weight = constraints.get('min_weight', 0.01)
        
        cov_df = pd.DataFrame(cov_matrix)
        dist = ((1 - corr_matrix) / 2.0) ** 0.5
        dist = np.nan_to_num(dist)
        
        link = linkage(dist[np.triu_indices(cov_matrix.shape[0], k=1)], method='single')
        sort_ix = self.get_quasi_diag(link)
        weights = self.get_rec_bipart(cov_df, sort_ix).values.astype(float)
        
        weights[weights < min_weight] = 0.0
        weights /= (weights.sum() + 1e-8)
        
        return weights


class HybridEnsembleExpert:
    """
    Hybrid ensemble that explicitly uses Risk Score to shift strategy.
    """
    
    def __init__(self, ensemble_weights=None, randomize_params=True, risk_profile=None):
        self.risk_profile = risk_profile or {}
        self.randomize_params = randomize_params
        self.risk_score = float(self.risk_profile.get('risk_score', 0.5))
        self.max_weight = self.risk_profile.get('max_weight', 0.3)
        self.min_weight = self.risk_profile.get('min_weight', 0.01)

        # === DYNAMIC STRATEGY ALLOCATION ===
        rho = self.risk_score
        conservative = 1.0 - rho
        
        dynamic_weights = {
            'robust_markowitz': 0.10 + 0.70 * rho,       # Aggressive favor
            'black_litterman':  0.10 + 0.10 * rho,       
            'risk_parity':      0.10 + 0.50 * conservative, # Conservative favor
            'hrp':              0.10 + 0.20 * conservative 
        }
        
        total = sum(dynamic_weights.values())
        self.ensemble_weights = {k: v/total for k, v in dynamic_weights.items()}
        
        if ensemble_weights:
            print("Info: Overriding passed ensemble_weights with risk_score derived weights.")

    def _get_randomized_params(self, method):
        """
        Get params. If randomize_params is False, use Deterministic Risk-Based Params.
        """
        # Base Risk Aversion: Conservative (0.0) -> 3.0, Aggressive (1.0) -> 1.0
        ra_base = 1.0 + (1.0 - self.risk_score) * 2.0
        
        if method == 'robust_markowitz':
            params = {
                'risk_aversion': ra_base,
                'regularization': 0.01,
                'shrinkage_method': 'ledoit_wolf'
            }
            if self.randomize_params:
                params['risk_aversion'] *= np.random.uniform(0.8, 1.2)
                params['regularization'] = np.random.uniform(0.001, 0.05)
            return params
            
        elif method == 'black_litterman':
            params = {
                'tau': 0.05,
                'risk_aversion': ra_base
            }
            if self.randomize_params:
                params['tau'] = np.random.uniform(0.01, 0.1)
                params['risk_aversion'] *= np.random.uniform(0.8, 1.2)
            return params
            
        return {}
    
    def generate_expert_action(self, returns, correlation_matrix, industry_matrix,
                              pos_matrix=None, neg_matrix=None, constraints=None):
        n_assets = len(returns)
        
        final_constraints = constraints.copy() if constraints else {}
        final_constraints['max_weight'] = final_constraints.get('max_weight', self.max_weight)
        final_constraints['min_weight'] = final_constraints.get('min_weight', self.min_weight)
        
        experts = {
            'robust_markowitz': RobustMarkowitz(**self._get_randomized_params('robust_markowitz')),
            'black_litterman': BlackLitterman(**self._get_randomized_params('black_litterman')),
            'risk_parity': RiskParity(),
            'hrp': HRP()
        }
        
        weights_sum = np.zeros(n_assets)
        total_trust = 0.0
        
        for name, strategy in experts.items():
            trust = self.ensemble_weights.get(name, 0.0)
            if trust < 0.01: continue
            
            try:
                if name == 'black_litterman':
                    if len(returns.shape) == 1:
                        r2d = np.tile(returns, (5, 1)) + np.random.randn(5, n_assets)*0.01
                    else: r2d = returns
                    cov = np.cov(r2d.T)
                    w = strategy.optimize(returns, cov, correlation_matrix, industry_matrix, constraints=final_constraints)
                else:
                    w = strategy.optimize(returns, constraints=final_constraints)
                
                weights_sum += w * trust
                total_trust += trust
            except Exception as e:
                continue
                
        if total_trust > 0:
            final_weights = weights_sum / total_trust
        else:
            final_weights = np.ones(n_assets) / n_assets
            
        return final_weights


# ... (Keep generate_expert_trajectories, save/load functions exactly as they were) ...
def generate_expert_trajectories(args, dataset, num_trajectories=100, risk_profile=None):
    """
    Generate expert trajectories using hybrid ensemble
    Returns:
        List of (state, weights) tuples
    """
    expert_trajectories = []
    
    print(f"\nGenerating {num_trajectories} Ensemble expert trajectories")
    print(f"Ensemble strategy: Hybrid (Markowitz + BL + RP + HRP)")
    print("="*60)
    if risk_profile:
        print(f"Risk-aware expert profile: score={risk_profile.get('risk_score', 0.5):.2f}, "
              f"max_weight={risk_profile.get('max_weight', 0.3):.2f}")
    
    # Initialize hybrid ensemble expert
    expert = HybridEnsembleExpert(randomize_params=True, risk_profile=risk_profile)
    
    for traj_idx in range(num_trajectories):
        # Randomly select a data point
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]
        
        # Extract features
        features = data['features'].numpy()
        returns = data['labels'].numpy()
        correlation_matrix = data['corr'].numpy()
        industry_matrix = data['industry_matrix'].numpy()
        pos_matrix = data['pos_matrix'].numpy() if 'pos_matrix' in data else None
        neg_matrix = data['neg_matrix'].numpy() if 'neg_matrix' in data else None
        
        n_stocks = len(returns)
        
        # Set constraints
        constraints = {
            'max_weight': risk_profile.get('max_weight', 0.3) if risk_profile else 0.3,
            'min_weight': risk_profile.get('min_weight', 0.01) if risk_profile else 0.01,
        }
        
        try:
            # Generate expert weights
            expert_weights = expert.generate_expert_action(
                returns=returns,
                correlation_matrix=correlation_matrix,
                industry_matrix=industry_matrix,
                pos_matrix=pos_matrix,
                neg_matrix=neg_matrix,
                constraints=constraints
            )
        except Exception as e:
            print(f"Warning: Trajectory {traj_idx} failed: {e}")
            # Fallback: simple return-weighted
            expert_weights = np.maximum(returns, 0)
            expert_weights = expert_weights / (expert_weights.sum() + 1e-8)
        
        # Construct state (flattened)
        state_parts = []
        if args.ind_yn:
            state_parts.append(industry_matrix.flatten())
        else:
            state_parts.append(np.zeros(industry_matrix.size))
        if args.pos_yn and pos_matrix is not None:
            state_parts.append(pos_matrix.flatten())
        else:
            state_parts.append(np.zeros(n_stocks * n_stocks))
        if args.neg_yn and neg_matrix is not None:
            state_parts.append(neg_matrix.flatten())
        else:
            state_parts.append(np.zeros(n_stocks * n_stocks))
        
        state_parts.append(features.flatten())
        state = np.concatenate(state_parts)
        
        expert_trajectories.append((state, expert_weights))  # ← weights
        
        if (traj_idx + 1) % 100 == 0:
            print(f"  Generated {traj_idx + 1}/{num_trajectories} trajectories")
    
    print(f"\nGenerated {len(expert_trajectories)} trajectories")
    
    # Statistics for weights
    all_weights = [a for _, a in expert_trajectories]
    avg_weight_sum = np.mean([np.sum(w) for w in all_weights])
    avg_nonzero = np.mean([np.sum(w > 0.01) for w in all_weights])
    max_concentration = np.mean([np.max(w) for w in all_weights])
    avg_entropy = np.mean([-np.sum(w * np.log(w + 1e-8)) for w in all_weights])
    
    print(f"  Weight statistics:")
    print(f"    - Avg stocks with >1% allocation: {avg_nonzero:.1f}")
    print(f"    - Avg max single stock weight: {max_concentration:.1%}")
    print(f"    - Avg portfolio entropy: {avg_entropy:.2f}")
    
    # Show sample weights
    sample_weights = all_weights[0]
    top_5_idx = np.argsort(-sample_weights)[:5]
    print(f"  Sample portfolio (first trajectory):")
    for i, idx in enumerate(top_5_idx):
        print(f"    Stock {idx}: {sample_weights[idx]:.3%}")
    
    return expert_trajectories


def save_expert_trajectories(trajectories, save_path):
    """Save expert trajectories to file"""
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {save_path}")


def load_expert_trajectories(load_path):
    """Load expert trajectories from file"""
    with open(load_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


if __name__ == '__main__':
    print("Testing Hybrid Ensemble Expert Strategy...")
    
    # Create dummy data
    n_stocks = 50
    
    returns = np.random.randn(n_stocks) * 0.01
    correlation_matrix = np.random.rand(n_stocks, n_stocks)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    industry_matrix = (np.random.rand(n_stocks, n_stocks) > 0.7).astype(float)
    
    # Initialize expert
    expert = HybridEnsembleExpert(randomize_params=False)
    
    weights = expert.generate_expert_action(
        returns=returns,
        correlation_matrix=correlation_matrix,
        industry_matrix=industry_matrix
    )
    print("Generated expert weights:")
    print(f"  Shape: {weights.shape}")
    print(f"  Sum: {weights.sum():.6f} (should be 1.0)")
    print(f"  Max weight: {weights.max():.3%}")
    print(f"  Stocks with >1%: {np.sum(weights > 0.01)}")
    print(f"  Top 5 allocations:")
    top_5_idx = np.argsort(-weights)[:5]
    for idx in top_5_idx:
        print(f"    Stock {idx}: {weights[idx]:.3%}")
    
    print("\nHybrid Ensemble Expert implementation complete!")