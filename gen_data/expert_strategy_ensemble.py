"""
Hybrid Ensemble Expert Strategy for Portfolio Optimization
Combines multiple advanced portfolio construction methods with continuous weights
"""

import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from scipy.linalg import sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, some optimization methods will be disabled")


class HybridEnsembleExpert:
    """
    Ensemble of advanced portfolio optimization strategies
    Outputs continuous portfolio weights (not binary selection)
    """
    
    def __init__(self, 
                 ensemble_weights: Optional[Dict[str, float]] = None,
                 randomize_params: bool = True,
                 min_weight: float = 0.01,
                 max_weight: float = 0.30):
        """
        Args:
            ensemble_weights: Dict of weights for each strategy. If None, uses balanced weights.
            randomize_params: Whether to randomize strategy parameters for diversity
            min_weight: Minimum allocation per stock (for selected stocks)
            max_weight: Maximum allocation per stock
        """
        self.randomize_params = randomize_params
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Default ensemble weights (can be customized)
        if ensemble_weights is None:
            self.ensemble_weights = {
                'mean_variance': 0.30,      # Classic Markowitz
                'risk_parity': 0.25,        # Equal risk contribution
                'min_variance': 0.20,       # Minimum volatility
                'max_sharpe': 0.15,         # Maximum Sharpe ratio
                'equal_weight': 0.10,       # Naive diversification
            }
        else:
            self.ensemble_weights = ensemble_weights
            
        # Normalize weights
        total = sum(self.ensemble_weights.values())
        self.ensemble_weights = {k: v/total for k, v in self.ensemble_weights.items()}
    
    def generate_expert_action(self,
                              returns: np.ndarray,
                              correlation_matrix: np.ndarray,
                              industry_matrix: Optional[np.ndarray] = None,
                              risk_aversion: float = 2.0) -> np.ndarray:
        """
        Generate continuous portfolio weights using ensemble of strategies
        
        Args:
            returns: Expected returns for each stock [n_stocks]
            correlation_matrix: Correlation matrix [n_stocks, n_stocks]
            industry_matrix: Industry relationship matrix (optional)
            risk_aversion: Risk aversion parameter (higher = more conservative)
            
        Returns:
            Continuous weights [n_stocks] that sum to 1.0
        """
        n_stocks = len(returns)
        
        # Randomize risk aversion if enabled
        if self.randomize_params:
            risk_aversion = np.random.uniform(1.0, 5.0)
        
        # Compute covariance from correlation and estimate volatilities
        volatilities = np.abs(returns) + 0.01  # Proxy: higher returns = higher vol
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Ensure positive definite
        cov_matrix = self._make_positive_definite(cov_matrix)
        
        # Generate weights from each strategy
        strategy_weights = {}
        
        if 'mean_variance' in self.ensemble_weights:
            strategy_weights['mean_variance'] = self._mean_variance_optimization(
                returns, cov_matrix, risk_aversion
            )
        
        if 'risk_parity' in self.ensemble_weights:
            strategy_weights['risk_parity'] = self._risk_parity_optimization(cov_matrix)
        
        if 'min_variance' in self.ensemble_weights:
            strategy_weights['min_variance'] = self._minimum_variance_optimization(cov_matrix)
        
        if 'max_sharpe' in self.ensemble_weights:
            strategy_weights['max_sharpe'] = self._max_sharpe_optimization(
                returns, cov_matrix
            )
        
        if 'equal_weight' in self.ensemble_weights:
            strategy_weights['equal_weight'] = self._equal_weight_portfolio(n_stocks)
        
        # Combine strategies using ensemble weights
        combined_weights = np.zeros(n_stocks)
        for strategy_name, weights in strategy_weights.items():
            if weights is not None:
                ensemble_w = self.ensemble_weights.get(strategy_name, 0.0)
                combined_weights += ensemble_w * weights
        
        # Apply constraints and normalize
        combined_weights = self._apply_constraints(combined_weights)
        
        return combined_weights
    
    def _make_positive_definite(self, matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Ensure matrix is positive definite"""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, epsilon)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def _mean_variance_optimization(self, 
                                   returns: np.ndarray, 
                                   cov_matrix: np.ndarray,
                                   risk_aversion: float) -> np.ndarray:
        """Classic Markowitz mean-variance optimization"""
        n = len(returns)
        
        if not SCIPY_AVAILABLE:
            # Fallback: return-weighted with risk adjustment
            adjusted_returns = returns - risk_aversion * np.diag(cov_matrix)
            weights = np.maximum(adjusted_returns, 0)
            return weights / (weights.sum() + 1e-8)
        
        # Objective: minimize -returns + risk_aversion * variance
        def objective(w):
            portfolio_return = w @ returns
            portfolio_variance = w @ cov_matrix @ w
            return -portfolio_return + risk_aversion * portfolio_variance
        
        # Constraints: weights sum to 1, all non-negative
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.max_weight) for _ in range(n)]
        
        # Initial guess: equal weights
        w0 = np.ones(n) / n
        
        try:
            result = minimize(objective, w0, method='SLSQP', 
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 100})
            if result.success:
                return result.x
        except:
            pass
        
        # Fallback
        weights = np.maximum(returns, 0)
        return weights / (weights.sum() + 1e-8)
    
    def _risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk Parity: equal risk contribution from each asset"""
        n = cov_matrix.shape[0]
        
        if not SCIPY_AVAILABLE:
            # Fallback: inverse volatility weighting
            volatilities = np.sqrt(np.diag(cov_matrix))
            weights = 1.0 / (volatilities + 1e-8)
            return weights / weights.sum()
        
        # Target: equal marginal risk contribution
        def risk_budget_objective(w):
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w / (portfolio_vol + 1e-8)
            risk_contrib = w * marginal_contrib
            # Minimize variance of risk contributions
            return np.var(risk_contrib)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.max_weight) for _ in range(n)]
        w0 = np.ones(n) / n
        
        try:
            result = minimize(risk_budget_objective, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 100})
            if result.success:
                return result.x
        except:
            pass
        
        # Fallback: inverse volatility
        volatilities = np.sqrt(np.diag(cov_matrix))
        weights = 1.0 / (volatilities + 1e-8)
        return weights / weights.sum()
    
    def _minimum_variance_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimum Variance Portfolio"""
        n = cov_matrix.shape[0]
        
        if not SCIPY_AVAILABLE:
            # Analytical solution exists for min variance
            try:
                inv_cov = np.linalg.inv(cov_matrix + np.eye(n) * 1e-6)
                ones = np.ones(n)
                weights = inv_cov @ ones / (ones @ inv_cov @ ones)
                weights = np.maximum(weights, 0)
                return weights / weights.sum()
            except:
                return np.ones(n) / n
        
        def objective(w):
            return w @ cov_matrix @ w
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.max_weight) for _ in range(n)]
        w0 = np.ones(n) / n
        
        try:
            result = minimize(objective, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 100})
            if result.success:
                return result.x
        except:
            pass
        
        return np.ones(n) / n
    
    def _max_sharpe_optimization(self, 
                                returns: np.ndarray, 
                                cov_matrix: np.ndarray,
                                risk_free_rate: float = 0.0) -> np.ndarray:
        """Maximum Sharpe Ratio Portfolio"""
        n = len(returns)
        
        if not SCIPY_AVAILABLE:
            # Heuristic: return/volatility weighted
            volatilities = np.sqrt(np.diag(cov_matrix))
            sharpe_proxy = returns / (volatilities + 1e-8)
            weights = np.maximum(sharpe_proxy, 0)
            return weights / (weights.sum() + 1e-8)
        
        # Convert to minimization problem
        def negative_sharpe(w):
            portfolio_return = w @ returns
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            return -(portfolio_return - risk_free_rate) / (portfolio_vol + 1e-8)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.max_weight) for _ in range(n)]
        w0 = np.ones(n) / n
        
        try:
            result = minimize(negative_sharpe, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 100})
            if result.success:
                return result.x
        except:
            pass
        
        # Fallback
        weights = np.maximum(returns, 0)
        return weights / (weights.sum() + 1e-8)
    
    def _equal_weight_portfolio(self, n_stocks: int) -> np.ndarray:
        """Naive 1/N portfolio"""
        return np.ones(n_stocks) / n_stocks
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply min/max constraints and normalize"""
        # Remove very small weights
        weights[weights < self.min_weight] = 0
        
        # Clip to max weight
        weights = np.clip(weights, 0, self.max_weight)
        
        # Renormalize
        weights = weights / (weights.sum() + 1e-8)
        
        return weights


def generate_expert_trajectories(args,
                                dataset,
                                num_trajectories: int = 100,
                                ensemble_weights: Optional[Dict[str, float]] = None,
                                randomize: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate expert trajectories using hybrid ensemble strategy
    
    Args:
        args: Arguments object with market, input_dim, ind_yn, pos_yn, neg_yn
        dataset: Dataset containing state and return information
        num_trajectories: Number of trajectories to generate
        ensemble_weights: Custom ensemble weights (None = balanced)
        randomize: Whether to randomize parameters for diversity
        
    Returns:
        List of (state, continuous_weights) tuples
    """
    expert_trajectories = []
    
    print(f"\nGenerating {num_trajectories} Ensemble expert trajectories")
    print(f"Ensemble strategy: Hybrid (MV + RP + MinVar + MaxSharpe + EW)")
    print("="*60)
    
    expert = HybridEnsembleExpert(
        ensemble_weights=ensemble_weights,
        randomize_params=randomize
    )
    
    for traj_idx in range(num_trajectories):
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]
        
        features = data['features'].numpy()
        correlation_matrix = data['corr'].numpy()
        ind_matrix = data['industry_matrix'].numpy()
        pos_matrix = data['pos_matrix'].numpy()
        neg_matrix = data['neg_matrix'].numpy()
        returns = data['labels'].numpy()
        
        try:
            # Generate continuous weights from ensemble
            expert_weights = expert.generate_expert_action(
                returns=returns,
                correlation_matrix=correlation_matrix,
                industry_matrix=ind_matrix,
                risk_aversion=np.random.uniform(1.5, 4.0) if randomize else 2.0
            )
        except Exception as e:
            print(f"Warning: Trajectory {traj_idx} failed: {e}")
            # Fallback: simple return-weighted portfolio
            n_select = min(15, len(returns))
            expert_weights = np.zeros(len(returns), dtype=np.float32)
            top_indices = np.argsort(-returns)[:n_select]
            top_returns = returns[top_indices]
            if top_returns.min() < 0:
                top_returns = top_returns - top_returns.min() + 0.01
            expert_weights[top_indices] = top_returns / top_returns.sum()
        
        # Create flattened state
        state_parts = []
        if args.ind_yn:
            state_parts.append(ind_matrix.flatten())
        else:
            state_parts.append(np.zeros(ind_matrix.size))
        if args.pos_yn:
            state_parts.append(pos_matrix.flatten())
        else:
            state_parts.append(np.zeros(pos_matrix.size))
        if args.neg_yn:
            state_parts.append(neg_matrix.flatten())
        else:
            state_parts.append(np.zeros(neg_matrix.size))
        
        state_parts.append(features.flatten())
        state = np.concatenate(state_parts)
        
        expert_trajectories.append((state, expert_weights))
        
        if (traj_idx + 1) % 100 == 0:
            print(f"  Generated {traj_idx + 1}/{num_trajectories} trajectories")
    
    print(f"\n✓ Generated {len(expert_trajectories)} trajectories")
    
    # Statistics for continuous weights
    all_weights = [a for _, a in expert_trajectories]
    avg_weight_sum = np.mean([np.sum(w) for w in all_weights])
    avg_nonzero = np.mean([np.sum(w > 0.01) for w in all_weights])
    max_concentration = np.mean([np.max(w) for w in all_weights])
    avg_entropy = np.mean([-np.sum(w * np.log(w + 1e-8)) for w in all_weights])
    
    print(f"  Weight statistics:")
    print(f"    - Average sum: {avg_weight_sum:.4f} (should be ~1.0)")
    print(f"    - Avg stocks with >1% allocation: {avg_nonzero:.1f}")
    print(f"    - Avg max single stock weight: {max_concentration:.1%}")
    print(f"    - Avg portfolio entropy: {avg_entropy:.2f} (higher = more diversified)")
    
    # Show sample weights
    sample_weights = all_weights[0]
    top_5_idx = np.argsort(-sample_weights)[:5]
    print(f"  Sample portfolio (first trajectory):")
    for i, idx in enumerate(top_5_idx):
        print(f"    Stock {idx}: {sample_weights[idx]:.3%}")
    
    return expert_trajectories


def save_expert_trajectories(trajectories: List, save_path: str):
    """Save expert trajectories to file"""
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {save_path}")


def load_expert_trajectories(load_path: str) -> List:
    """Load expert trajectories from file"""
    with open(load_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories
