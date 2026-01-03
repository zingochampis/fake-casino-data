"""
Casino Gaming Data Generator for Predictive Analytics POC
Generates realistic gaming transaction data with Pareto distributions,
whale behavior, and temporal patterns based on academic research.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class CasinoDataGenerator:
    """Generate realistic casino gaming data with research-based distributions"""
    
    def __init__(self, output_dir: str = r"C:\Users\DmitriApassov\Desktop\POC"):
        """Initialize the generator with output directory"""
        self.output_dir = output_dir
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize game metadata
        self.games = self._initialize_games()
        
        # Initialize player segments based on research
        self.num_players = 10000
        self.players = self._initialize_players()
        
    def _initialize_games(self) -> Dict:
        """Initialize 4 games with different characteristics"""
        games = {
            'SLOT_FORTUNE': {
                'game_id': 'G001',
                'type': 'slot',
                'rtp': 0.96,  # 96% RTP
                'volatility': 'high',  # High volatility = rare big wins
                'min_bet': 0.20,
                'max_bet': 500,
                'hit_frequency': 0.25,  # 25% of spins win something
                'max_multiplier': 5000,
                'bonus_frequency': 0.008,  # 0.8% chance of bonus
                'visual_engagement': 0.85  # High engagement score
            },
            'BLACKJACK_LIVE': {
                'game_id': 'G002',
                'type': 'table',
                'rtp': 0.995,  # 99.5% RTP with perfect play
                'volatility': 'low',
                'min_bet': 5,
                'max_bet': 2000,
                'hit_frequency': 0.42,  # Win ~42% of hands
                'max_multiplier': 2.5,  # Blackjack pays 3:2
                'bonus_frequency': 0,
                'visual_engagement': 0.70
            },
            'ROULETTE_EURO': {
                'game_id': 'G003',
                'type': 'table',
                'rtp': 0.973,  # 97.3% RTP (single zero)
                'volatility': 'medium',
                'min_bet': 1,
                'max_bet': 1000,
                'hit_frequency': 0.486,  # 18/37 for even money bets
                'max_multiplier': 35,  # Straight up pays 35:1
                'bonus_frequency': 0,
                'visual_engagement': 0.65
            },
            'SLOT_CLASSIC': {
                'game_id': 'G004',
                'type': 'slot',
                'rtp': 0.94,  # Lower RTP classic slot
                'volatility': 'medium',
                'min_bet': 0.10,
                'max_bet': 100,
                'hit_frequency': 0.35,
                'max_multiplier': 1000,
                'bonus_frequency': 0.015,
                'visual_engagement': 0.60
            }
        }
        return games
    
    def _initialize_players(self) -> pd.DataFrame:
        """
        Initialize players with Pareto distribution for value segments
        Based on research: top 20% generate 80-90% of revenue
        """
        players = []
        
        # Generate player segments with power law distribution
        # Research shows approximately: 1% whales, 4% VIP, 10% high, 25% regular, 60% casual
        segments = {
            'whale': int(self.num_players * 0.01),      # 1% - Top spenders
            'vip': int(self.num_players * 0.04),        # 4% - VIP players  
            'high': int(self.num_players * 0.10),       # 10% - High value
            'regular': int(self.num_players * 0.25),    # 25% - Regular players
            'casual': int(self.num_players * 0.60)      # 60% - Casual players
        }
        
        player_id = 1
        for segment, count in segments.items():
            for _ in range(count):
                # Set player characteristics based on segment
                if segment == 'whale':
                    avg_bet = np.random.lognormal(5.5, 1.2)  # Mean ~$245, heavy tail
                    sessions_per_week = np.random.gamma(8, 1)  # 8+ sessions/week
                    avg_session_minutes = np.random.gamma(150, 20)  # 2.5+ hours
                    risk_tolerance = np.random.beta(8, 2)  # High risk tolerance
                    deposit_amount = np.random.lognormal(9, 1)  # ~$8000 avg deposit
                    
                elif segment == 'vip':
                    avg_bet = np.random.lognormal(3.5, 1.0)  # Mean ~$33
                    sessions_per_week = np.random.gamma(5, 1)
                    avg_session_minutes = np.random.gamma(90, 15)
                    risk_tolerance = np.random.beta(6, 4)
                    deposit_amount = np.random.lognormal(7, 0.8)  # ~$1100 avg
                    
                elif segment == 'high':
                    avg_bet = np.random.lognormal(2.3, 0.8)  # Mean ~$10
                    sessions_per_week = np.random.gamma(3, 1)
                    avg_session_minutes = np.random.gamma(60, 10)
                    risk_tolerance = np.random.beta(5, 5)
                    deposit_amount = np.random.lognormal(5.5, 0.7)  # ~$245 avg
                    
                elif segment == 'regular':
                    avg_bet = np.random.lognormal(1.0, 0.6)  # Mean ~$2.7
                    sessions_per_week = np.random.gamma(2, 0.5)
                    avg_session_minutes = np.random.gamma(40, 8)
                    risk_tolerance = np.random.beta(4, 6)
                    deposit_amount = np.random.lognormal(4, 0.6)  # ~$55 avg
                    
                else:  # casual
                    avg_bet = np.random.lognormal(0, 0.5)  # Mean ~$1
                    sessions_per_week = np.random.gamma(1, 0.3)
                    avg_session_minutes = np.random.gamma(20, 5)
                    risk_tolerance = np.random.beta(2, 8)
                    deposit_amount = np.random.lognormal(3, 0.5)  # ~$20 avg
                
                # Preferred games based on segment
                if segment in ['whale', 'vip']:
                    # High rollers prefer table games and high volatility slots
                    game_preference = np.random.choice(
                        list(self.games.keys()),
                        p=[0.35, 0.30, 0.25, 0.10]  # Prefer high stakes games
                    )
                else:
                    # Casual players prefer slots
                    game_preference = np.random.choice(
                        list(self.games.keys()),
                        p=[0.40, 0.10, 0.15, 0.35]  # Prefer slots
                    )
                
                # Churn risk based on research (higher for casual players)
                if segment == 'casual':
                    churn_probability = np.random.beta(2, 5)  # Higher churn risk
                elif segment == 'regular':
                    churn_probability = np.random.beta(2, 8)
                else:
                    churn_probability = np.random.beta(1, 20)  # Low churn for VIPs
                
                # Create player profile
                players.append({
                    'player_id': f'P{player_id:06d}',
                    'segment': segment,
                    'avg_bet': max(0.10, avg_bet),
                    'sessions_per_week': max(0.5, sessions_per_week),
                    'avg_session_minutes': max(5, avg_session_minutes),
                    'preferred_game': game_preference,
                    'risk_tolerance': risk_tolerance,
                    'deposit_amount': max(10, deposit_amount),
                    'churn_probability': churn_probability,
                    'registration_date': self.start_date + timedelta(days=np.random.randint(0, 365)),
                    'is_active': True
                })
                player_id += 1
        
        return pd.DataFrame(players)
    
    def _generate_session_times(self, player: pd.Series, week_start: datetime) -> List[datetime]:
        """
        Generate realistic session start times for a player in a given week
        Including weekend patterns and time-of-day preferences
        """
        sessions = []
        num_sessions = np.random.poisson(player['sessions_per_week'])
        
        for _ in range(num_sessions):
            # Day of week (0=Monday, 6=Sunday)
            # Research shows increased activity Fri-Sun
            day_weights = [0.10, 0.10, 0.12, 0.13, 0.20, 0.20, 0.15]  # Mon-Sun
            
            # Payday effect - increase weights for 1st and 15th
            current_day = week_start.day
            if 1 <= current_day <= 3 or 14 <= current_day <= 17:
                day_weights = [w * 1.3 for w in day_weights]  # 30% boost
            
            # Normalize weights
            day_weights = np.array(day_weights) / sum(day_weights)
            day = np.random.choice(7, p=day_weights)
            
            # Time of day - peak hours 8 PM - 2 AM
            # Using mixture of gaussians for realistic time distribution
            if np.random.random() < 0.7:  # 70% evening sessions
                hour = int(np.random.normal(21, 2)) % 24  # Peak at 9 PM
            else:  # 30% other times
                hour = np.random.randint(0, 24)
            
            minute = np.random.randint(0, 60)
            
            session_time = week_start + timedelta(days=day, hours=hour, minutes=minute)
            sessions.append(session_time)
        
        return sorted(sessions)
    
    def _simulate_game_session(self, player: pd.Series, game: Dict, 
                              session_start: datetime, session_id: str) -> List[Dict]:
        """
        Simulate a single gaming session with realistic betting patterns
        Including loss chasing and hot hand behaviors from research
        """
        transactions = []
        
        # Session duration with some randomness
        session_minutes = max(1, np.random.normal(
            player['avg_session_minutes'],
            player['avg_session_minutes'] * 0.3
        ))
        
        current_time = session_start
        session_balance = 0
        consecutive_losses = 0
        consecutive_wins = 0
        
        # Initial bet size
        base_bet = player['avg_bet']
        current_bet = base_bet
        
        # Generate transactions during session
        while (current_time - session_start).total_seconds() < session_minutes * 60:
            # Place bet
            bet_amount = min(current_bet, game['max_bet'])
            bet_amount = max(bet_amount, game['min_bet'])
            
            # Determine win/loss based on game mechanics
            if np.random.random() < game['hit_frequency']:
                # Win
                if game['type'] == 'slot':
                    # Slot wins follow power law distribution
                    if np.random.random() < game['bonus_frequency']:
                        # Bonus win - big multiplier
                        multiplier = np.random.pareto(1.5) * 10
                        multiplier = min(multiplier, game['max_multiplier'])
                    else:
                        # Regular win
                        multiplier = np.random.pareto(2.5) + 1
                        multiplier = min(multiplier, game['max_multiplier'] * 0.1)
                else:
                    # Table game wins are more consistent
                    if game['game_id'] == 'G002':  # Blackjack
                        multiplier = np.random.choice([1, 1.5], p=[0.8, 0.2])
                    else:  # Roulette
                        multiplier = np.random.choice([1, 2, 3, 5, 8, 11, 17, 35],
                                                    p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.015, 0.005])
                
                win_amount = bet_amount * multiplier
                consecutive_wins += 1
                consecutive_losses = 0
                
                # Hot hand effect - slightly increase bet after wins
                if consecutive_wins >= 2 and player['risk_tolerance'] > 0.5:
                    current_bet = min(current_bet * 1.1, base_bet * 2)
                
            else:
                # Loss
                win_amount = 0
                consecutive_losses += 1
                consecutive_wins = 0
                
                # Loss chasing behavior (research-based)
                if consecutive_losses >= 3 and player['risk_tolerance'] > 0.6:
                    # Chase losses by increasing bet
                    if player['segment'] in ['whale', 'vip']:
                        current_bet = min(current_bet * 1.5, base_bet * 5)
                    else:
                        current_bet = min(current_bet * 1.2, base_bet * 2)
                elif consecutive_losses >= 5:
                    # Cool down after many losses
                    current_bet = base_bet
            
            # Update session balance
            session_balance += (win_amount - bet_amount)
            
            # Create transaction record
            transactions.append({
                'transaction_id': f"{session_id}_{len(transactions):04d}",
                'player_id': player['player_id'],
                'game_id': game['game_id'],
                'timestamp': current_time,
                'bet_amount': round(bet_amount, 2),
                'win_amount': round(win_amount, 2),
                'net_result': round(win_amount - bet_amount, 2),
                'session_id': session_id,
                'session_balance': round(session_balance, 2),
                'consecutive_losses': consecutive_losses,
                'consecutive_wins': consecutive_wins
            })
            
            # Time until next bet (faster for slots, slower for table games)
            if game['type'] == 'slot':
                seconds_to_next = np.random.exponential(5)  # ~5 seconds average
            else:
                seconds_to_next = np.random.exponential(30)  # ~30 seconds average
            
            current_time += timedelta(seconds=seconds_to_next)
            
            # Small chance to end session early if big loss/win
            if session_balance < -player['deposit_amount'] * 0.5:
                if np.random.random() < 0.3:  # 30% chance to quit after big loss
                    break
            elif session_balance > player['deposit_amount'] * 2:
                if np.random.random() < 0.2:  # 20% chance to quit after big win
                    break
        
        return transactions
    
    def generate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate full year of gaming data
        Returns: transactions_df, daily_summary_df, player_summary_df
        """
        print("Starting data generation for casino POC...")
        print(f"Generating data for {self.num_players} players across 4 games")
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        
        all_transactions = []
        session_counter = 1
        
        # Calculate total weeks in the date range
        total_days = (self.end_date - self.start_date).days + 1
        total_weeks = (total_days + 6) // 7  # Round up to nearest week
        
        # Generate week by week
        current_date = self.start_date
        week_num = 0
        
        while current_date <= self.end_date:
            week_num += 1
            week_start = current_date
            week_end = min(current_date + timedelta(days=7), self.end_date)
            
            if week_num % 4 == 0:
                print(f"Processing week {week_num}/{total_weeks}...")
            
            # For each active player
            for _, player in self.players.iterrows():
                # Skip if player hasn't registered yet
                if player['registration_date'] > week_end:
                    continue
                
                # Simulate churn (player becomes inactive)
                if player['is_active']:
                    if np.random.random() < player['churn_probability'] / 52:  # Weekly churn chance
                        player['is_active'] = False
                        continue
                
                # Generate sessions for this week
                session_times = self._generate_session_times(player, week_start)
                
                for session_time in session_times:
                    if session_time > self.end_date:
                        continue
                    
                    # Choose game for this session (80% preferred, 20% other)
                    if np.random.random() < 0.8:
                        game_name = player['preferred_game']
                    else:
                        game_name = np.random.choice(list(self.games.keys()))
                    
                    game = self.games[game_name]
                    session_id = f"S{session_counter:08d}"
                    session_counter += 1
                    
                    # Generate session transactions
                    session_transactions = self._simulate_game_session(
                        player, game, session_time, session_id
                    )
                    all_transactions.extend(session_transactions)
            
            # Move to next week (avoid infinite loop at the end)
            if week_end >= self.end_date:
                break
            current_date = week_end + timedelta(days=1)
        
        # Convert to DataFrame
        print(f"\nGenerated {len(all_transactions):,} transactions")
        transactions_df = pd.DataFrame(all_transactions)
        
        # Add derived features for analytics
        if not transactions_df.empty:
            transactions_df['hour'] = pd.to_datetime(transactions_df['timestamp']).dt.hour
            transactions_df['day_of_week'] = pd.to_datetime(transactions_df['timestamp']).dt.dayofweek
            transactions_df['day_of_month'] = pd.to_datetime(transactions_df['timestamp']).dt.day
            transactions_df['month'] = pd.to_datetime(transactions_df['timestamp']).dt.month
            transactions_df['is_weekend'] = transactions_df['day_of_week'].isin([5, 6]).astype(int)
            transactions_df['is_peak_hour'] = transactions_df['hour'].between(20, 23).astype(int)
            transactions_df['is_payday'] = transactions_df['day_of_month'].isin([1, 15]).astype(int)
        
        # Create daily summary
        print("Creating daily summaries...")
        daily_summary = self._create_daily_summary(transactions_df)
        
        # Create player summary
        print("Creating player summaries...")
        player_summary = self._create_player_summary(transactions_df)
        
        return transactions_df, daily_summary, player_summary
    
    def _create_daily_summary(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create daily aggregated metrics"""
        if transactions_df.empty:
            return pd.DataFrame()
        
        transactions_df['date'] = pd.to_datetime(transactions_df['timestamp']).dt.date
        
        daily_summary = transactions_df.groupby('date').agg({
            'transaction_id': 'count',
            'player_id': 'nunique',
            'bet_amount': 'sum',
            'win_amount': 'sum',
            'net_result': 'sum',
            'session_id': 'nunique'
        }).rename(columns={
            'transaction_id': 'total_bets',
            'player_id': 'active_players',
            'bet_amount': 'total_wagered',
            'win_amount': 'total_won',
            'net_result': 'gross_gaming_revenue',
            'session_id': 'total_sessions'
        })
        
        # Add derived metrics
        daily_summary['hold_percentage'] = (
            -daily_summary['gross_gaming_revenue'] / daily_summary['total_wagered'] * 100
        )
        daily_summary['avg_bet_size'] = (
            daily_summary['total_wagered'] / daily_summary['total_bets']
        )
        daily_summary['bets_per_player'] = (
            daily_summary['total_bets'] / daily_summary['active_players']
        )
        
        return daily_summary.reset_index()
    
    def _create_player_summary(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create player lifetime value summary"""
        if transactions_df.empty:
            return pd.DataFrame()
        
        player_summary = transactions_df.groupby('player_id').agg({
            'transaction_id': 'count',
            'bet_amount': ['sum', 'mean', 'std'],
            'win_amount': 'sum',
            'net_result': 'sum',
            'session_id': 'nunique',
            'timestamp': ['min', 'max']
        })
        
        # Flatten column names
        player_summary.columns = ['_'.join(col).strip() for col in player_summary.columns.values]
        player_summary = player_summary.rename(columns={
            'transaction_id_count': 'total_bets',
            'bet_amount_sum': 'total_wagered',
            'bet_amount_mean': 'avg_bet',
            'bet_amount_std': 'bet_volatility',
            'win_amount_sum': 'total_won',
            'net_result_sum': 'net_loss',
            'session_id_nunique': 'total_sessions',
            'timestamp_min': 'first_bet',
            'timestamp_max': 'last_bet'
        })
        
        # Add player segment from original data
        player_summary = player_summary.reset_index().merge(
            self.players[['player_id', 'segment', 'preferred_game']],
            on='player_id',
            how='left'
        )
        
        # Calculate days active
        player_summary['days_active'] = (
            pd.to_datetime(player_summary['last_bet']) - 
            pd.to_datetime(player_summary['first_bet'])
        ).dt.days + 1
        
        # Calculate player value score (for VIP identification)
        player_summary['lifetime_value'] = -player_summary['net_loss']
        player_summary['avg_daily_value'] = (
            player_summary['lifetime_value'] / player_summary['days_active']
        )
        
        return player_summary
    
    def save_data(self, transactions_df: pd.DataFrame, 
                  daily_summary: pd.DataFrame,
                  player_summary: pd.DataFrame):
        """Save data with memory optimization for large datasets"""
        print(f"\nSaving data to {self.output_dir}...")
        
        # Save Metadata and Summaries first (they are small)
        daily_summary.to_csv(os.path.join(self.output_dir, 'daily_summary.csv'), index=False)
        player_summary.to_csv(os.path.join(self.output_dir, 'player_summary.csv'), index=False)
        pd.DataFrame.from_dict(self.games, orient='index').to_csv(os.path.join(self.output_dir, 'games_metadata.csv'))

        # Handle the massive transactions table
        if not transactions_df.empty:
            # Create a simple string column for month to avoid Period object memory overhead
            transactions_df['temp_month'] = pd.to_datetime(transactions_df['timestamp']).dt.strftime('%Y-%m')
            
            months = transactions_df['temp_month'].unique()
            for month in months:
                month_df = transactions_df[transactions_df['temp_month'] == month]
                filename = os.path.join(self.output_dir, f'transactions_{month}.csv')
                # Drop the helper column before saving
                month_df.drop(columns=['temp_month']).to_csv(filename, index=False)
                print(f"Saved {filename} ({len(month_df):,} rows)")
                
                # Manual hint to clear memory if needed
                del month_df
            
            transactions_df.drop(columns=['temp_month'], inplace=True)

        # Save stats
        stats = {
            'total_players': len(self.players),
            'total_transactions': len(transactions_df),
            'total_wagered': float(transactions_df['bet_amount'].sum()),
            'gross_gaming_revenue': float(-transactions_df['net_result'].sum()),
        }
        with open(os.path.join(self.output_dir, 'generation_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        print("\nData generation complete!")

        return stats

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("Casino Gaming Data Generator for Predictive Analytics POC")
    print("="*60)
    
    # Initialize generator
    generator = CasinoDataGenerator()
    
    # Generate data
    transactions, daily_summary, player_summary = generator.generate_data()
    
    # Save data
    generator.save_data(transactions, daily_summary, player_summary)
    
    print("\n" + "="*60)
    print("Data generation completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Open KNIME Analytics Platform")
    print("2. Import the CSV files from the POC folder")
    print("3. Build time series prediction models")
    print("4. Create visualizations and reports")
    print("\nKey files generated:")
    print("  - transactions_*.csv: Raw betting data")
    print("  - daily_summary.csv: Daily aggregated metrics")
    print("  - player_summary.csv: Player lifetime values")
    print("  - games_metadata.csv: Game characteristics")
    print("  - generation_stats.json: Generation statistics")