"""
Data Simulator for Kafka-based Fraud Detection
Simulates realistic transaction streams with configurable fraud rates
"""
import json
import logging
import time
import signal
import sys
import random
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
from kafka_config import kafka_manager, KafkaConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionSimulator:
    """Simulates realistic credit card transactions"""
    
    def __init__(self):
        self.config = KafkaConfig()
        self.running = False
        self.transaction_counter = 0
        
        # Simulation parameters
        self.transactions_per_second = 10  # Configurable throughput
        self.fraud_rate = 0.02  # 2% fraud rate
        self.burst_mode = False
        self.burst_multiplier = 5
        
    def generate_normal_transaction(self) -> Dict[str, Any]:
        """Generate a normal (non-fraudulent) transaction"""
        transaction_id = f"tx_{int(time.time() * 1000)}_{self.transaction_counter}"
        self.transaction_counter += 1
        
        # Generate realistic normal transaction features
        # Time feature (seconds from start of day)
        current_time = datetime.now()
        time_feature = (current_time.hour * 3600 + 
                       current_time.minute * 60 + 
                       current_time.second) + random.uniform(-300, 300)
        
        # Amount (normal distribution with typical spending patterns)
        amount = abs(np.random.lognormal(mean=3.5, sigma=1.2))  # ~$30-$300 typical
        amount = min(amount, 2000)  # Cap at $2000
        
        # V1-V28 features (PCA components - normal distribution)
        v_features = {}
        for i in range(1, 29):
            if i <= 14:
                # First 14 features tend to be smaller
                v_features[f'v{i}'] = np.random.normal(0, 1)
            else:
                # Last 14 features tend to be even smaller
                v_features[f'v{i}'] = np.random.normal(0, 0.5)
        
        transaction = {
            'transaction_id': transaction_id,
            'time_feature': time_feature,
            'amount': amount,
            'timestamp': current_time.isoformat(),
            'simulation_type': 'normal',
            'actual_class': 0,  # 0 = normal transaction
            **v_features
        }
        
        return transaction
    
    def generate_fraud_transaction(self) -> Dict[str, Any]:
        """Generate a fraudulent transaction with suspicious patterns"""
        transaction_id = f"fraud_tx_{int(time.time() * 1000)}_{self.transaction_counter}"
        self.transaction_counter += 1
        
        current_time = datetime.now()
        
        # Fraudulent patterns
        fraud_patterns = [
            'high_amount',
            'unusual_time',
            'rapid_succession',
            'unusual_location'
        ]
        
        pattern = random.choice(fraud_patterns)
        
        if pattern == 'high_amount':
            # Unusually high amount
            amount = random.uniform(5000, 25000)
            time_feature = (current_time.hour * 3600 + 
                           current_time.minute * 60 + 
                           current_time.second)
        elif pattern == 'unusual_time':
            # Transaction at unusual hours (2-5 AM)
            unusual_hour = random.choice([2, 3, 4])
            time_feature = unusual_hour * 3600 + random.uniform(0, 3600)
            amount = random.uniform(100, 1000)
        elif pattern == 'rapid_succession':
            # Quick successive transactions
            time_feature = (current_time.hour * 3600 + 
                           current_time.minute * 60 + 
                           current_time.second)
            amount = random.uniform(200, 800)
        else:  # unusual_location
            # Unusual location pattern (reflected in V features)
            time_feature = (current_time.hour * 3600 + 
                           current_time.minute * 60 + 
                           current_time.second)
            amount = random.uniform(50, 500)
        
        # Generate V features with fraud patterns
        v_features = {}
        for i in range(1, 29):
            if pattern == 'unusual_location' and i in [1, 2, 3, 9, 10, 14]:
                # Unusual location affects certain V features
                v_features[f'v{i}'] = np.random.normal(0, 3)  # Higher variance
            elif pattern == 'rapid_succession' and i in [4, 11, 12, 18]:
                # Rapid succession affects transaction behavior features
                v_features[f'v{i}'] = np.random.normal(2, 1)  # Shifted mean
            else:
                # Other features with slightly elevated variance
                base_std = 1.5 if i <= 14 else 0.8
                v_features[f'v{i}'] = np.random.normal(0, base_std)
        
        transaction = {
            'transaction_id': transaction_id,
            'time_feature': time_feature,
            'amount': amount,
            'timestamp': current_time.isoformat(),
            'simulation_type': 'fraud',
            'fraud_pattern': pattern,
            'actual_class': 1,  # 1 = fraud transaction
            **v_features
        }
        
        return transaction
    
    def simulate_transaction_stream(self, duration_minutes: Optional[int] = None):
        """Simulate continuous transaction stream"""
        logger.info(f"Starting transaction simulation at {self.transactions_per_second} TPS")
        
        start_time = time.time()
        
        try:
            while self.running:
                batch_start = time.time()
                
                # Determine current throughput
                current_tps = self.transactions_per_second
                if self.burst_mode:
                    current_tps *= self.burst_multiplier
                    logger.info(f"BURST MODE: {current_tps} TPS")
                
                # Generate batch of transactions
                transactions_this_second = int(current_tps)
                
                for _ in range(transactions_this_second):
                    if not self.running:
                        break
                    
                    # Determine if this should be fraud
                    is_fraud = random.random() < self.fraud_rate
                    
                    if is_fraud:
                        transaction = self.generate_fraud_transaction()
                    else:
                        transaction = self.generate_normal_transaction()
                    
                    # Send to Kafka
                    success = kafka_manager.send_message(
                        topic=self.config.topics['transactions'],
                        value=transaction,
                        key=transaction['transaction_id']
                    )
                    
                    if success:
                        logger.debug(f"Sent transaction: {transaction['transaction_id']} ({'FRAUD' if is_fraud else 'NORMAL'})")
                    else:
                        logger.error(f"Failed to send transaction: {transaction['transaction_id']}")
                
                # Random burst mode toggle
                if random.random() < 0.05:  # 5% chance per second
                    self.burst_mode = not self.burst_mode
                    if self.burst_mode:
                        logger.info("Entering burst mode (simulating high traffic)")
                    else:
                        logger.info("Exiting burst mode")
                
                # Sleep to maintain target TPS
                elapsed = time.time() - batch_start
                sleep_time = max(0, 1.0 - elapsed)
                time.sleep(sleep_time)
                
                # Log progress every minute
                if int(time.time() - start_time) % 60 == 0:
                    runtime = int(time.time() - start_time)
                    logger.info(f"Simulation running for {runtime}s. Generated ~{self.transaction_counter} transactions")
                
                # Check duration limit
                if duration_minutes and (time.time() - start_time) > (duration_minutes * 60):
                    logger.info(f"Simulation completed after {duration_minutes} minutes")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.stop()
    
    def start(self, duration_minutes: Optional[int] = None):
        """Start the transaction simulation"""
        self.running = True
        
        # Wait for Kafka to be ready
        logger.info("Waiting for Kafka to be ready...")
        max_retries = 30
        for i in range(max_retries):
            try:
                if kafka_manager.health_check():
                    logger.info("Kafka is ready!")
                    break
            except Exception as e:
                logger.warning(f"Kafka not ready (attempt {i+1}/{max_retries}): {e}")
                time.sleep(2)
        else:
            logger.error("Kafka not available after maximum retries")
            return
        
        # Create topics if needed
        from kafka_config import create_topics_if_not_exist
        create_topics_if_not_exist()
        
        # Start simulation
        self.simulate_transaction_stream(duration_minutes)
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
        kafka_manager.close_producer()
        logger.info("Transaction simulation stopped")
    
    def adjust_throughput(self, new_tps: int):
        """Dynamically adjust transactions per second"""
        old_tps = self.transactions_per_second
        self.transactions_per_second = max(1, min(1000, new_tps))  # Limit between 1-1000 TPS
        logger.info(f"Throughput adjusted from {old_tps} to {self.transactions_per_second} TPS")

class ConfigurableSimulator:
    """Advanced simulator with multiple configuration options"""
    
    def __init__(self):
        self.simulators = []
        self.running = False
        
    def add_simulator_profile(self, name: str, tps: int, fraud_rate: float):
        """Add a simulator with specific profile"""
        simulator = TransactionSimulator()
        simulator.transactions_per_second = tps
        simulator.fraud_rate = fraud_rate
        self.simulators.append({
            'name': name,
            'simulator': simulator,
            'thread': None
        })
        logger.info(f"Added simulator profile '{name}': {tps} TPS, {fraud_rate*100:.1f}% fraud rate")
    
    def start_all_simulators(self):
        """Start all configured simulators"""
        self.running = True
        
        for sim_config in self.simulators:
            simulator = sim_config['simulator']
            thread = threading.Thread(
                target=simulator.start,
                name=f"Simulator-{sim_config['name']}",
                daemon=True
            )
            sim_config['thread'] = thread
            thread.start()
            logger.info(f"Started simulator: {sim_config['name']}")
    
    def stop_all_simulators(self):
        """Stop all running simulators"""
        self.running = False
        
        for sim_config in self.simulators:
            sim_config['simulator'].stop()
            if sim_config['thread']:
                sim_config['thread'].join(timeout=5)
        
        logger.info("All simulators stopped")
    
    def stop(self):
        """Alias for stop_all_simulators for consistency"""
        self.stop_all_simulators()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    global main_simulator
    if main_simulator:
        if isinstance(main_simulator, ConfigurableSimulator):
            main_simulator.stop_all_simulators()
        elif isinstance(main_simulator, TransactionSimulator):
            main_simulator.stop()
    sys.exit(0)

# Global simulator instance
main_simulator = None

def main():
    """Main simulator process"""
    global main_simulator
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configuration from environment
    simulation_mode = os.getenv('SIMULATION_MODE', 'single')  # single, multi, burst
    base_tps = int(os.getenv('BASE_TPS', '10'))
    fraud_rate = float(os.getenv('FRAUD_RATE', '0.02'))
    
    if simulation_mode == 'multi':
        # Multi-profile simulation
        logger.info("Starting multi-profile simulation")
        main_simulator = ConfigurableSimulator()
        
        # Add different simulator profiles
        main_simulator.add_simulator_profile('normal', base_tps, fraud_rate)
        main_simulator.add_simulator_profile('high_volume', base_tps * 3, fraud_rate * 0.5)
        main_simulator.add_simulator_profile('fraud_spike', base_tps // 2, fraud_rate * 5)
        
        main_simulator.start_all_simulators()
        
        # Keep main thread alive
        try:
            while main_simulator.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            main_simulator.stop_all_simulators()
            
    else:
        # Single simulator
        logger.info(f"Starting single simulation at {base_tps} TPS")
        main_simulator = TransactionSimulator()
        main_simulator.transactions_per_second = base_tps
        main_simulator.fraud_rate = fraud_rate
        
        if simulation_mode == 'burst':
            main_simulator.burst_mode = True
            logger.info("Burst mode enabled")
        
        main_simulator.start()

if __name__ == '__main__':
    main()
