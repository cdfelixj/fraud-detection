#!/usr/bin/env python3
"""
Database migration script to make actual_class column nullable.
This allows the system to handle transactions without ground truth labels.

Run this script once to update your existing database.
"""

import os
import sys
from sqlalchemy import text

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import Transaction

def migrate_actual_class_column():
    """Make the actual_class column nullable in the transactions table"""
    
    with app.app_context():
        try:
            # Check if we're using SQLite or PostgreSQL
            engine_name = db.engine.name
            
            if engine_name == 'sqlite':
                # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
                print("Detected SQLite database. Recreating transactions table...")
                
                # Get existing data
                existing_transactions = db.session.execute(
                    text("SELECT * FROM transactions")
                ).fetchall()
                
                # Drop and recreate table
                db.session.execute(text("DROP TABLE IF EXISTS transactions_backup"))
                db.session.execute(text("""
                    CREATE TABLE transactions_backup AS 
                    SELECT * FROM transactions
                """))
                
                # Drop the existing table
                Transaction.__table__.drop(db.engine)
                
                # Create new table with nullable actual_class
                Transaction.__table__.create(db.engine)
                
                # Restore data
                if existing_transactions:
                    print(f"Restoring {len(existing_transactions)} transactions...")
                    for row in existing_transactions:
                        insert_sql = text("""
                            INSERT INTO transactions 
                            (id, transaction_id, time_feature, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, 
                             v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, 
                             v26, v27, v28, amount, actual_class, created_at)
                            VALUES 
                            (:id, :transaction_id, :time_feature, :v1, :v2, :v3, :v4, :v5, :v6, :v7, :v8, :v9, :v10,
                             :v11, :v12, :v13, :v14, :v15, :v16, :v17, :v18, :v19, :v20, :v21, :v22, :v23, :v24, :v25,
                             :v26, :v27, :v28, :amount, :actual_class, :created_at)
                        """)
                        
                        db.session.execute(insert_sql, {
                            'id': row[0],
                            'transaction_id': row[1],
                            'time_feature': row[2],
                            'v1': row[3], 'v2': row[4], 'v3': row[5], 'v4': row[6], 'v5': row[7],
                            'v6': row[8], 'v7': row[9], 'v8': row[10], 'v9': row[11], 'v10': row[12],
                            'v11': row[13], 'v12': row[14], 'v13': row[15], 'v14': row[16], 'v15': row[17],
                            'v16': row[18], 'v17': row[19], 'v18': row[20], 'v19': row[21], 'v20': row[22],
                            'v21': row[23], 'v22': row[24], 'v23': row[25], 'v24': row[26], 'v25': row[27],
                            'v26': row[28], 'v27': row[29], 'v28': row[30],
                            'amount': row[31],
                            'actual_class': row[32],
                            'created_at': row[33]
                        })
                
                # Clean up backup
                db.session.execute(text("DROP TABLE transactions_backup"))
                
            elif engine_name == 'postgresql':
                # PostgreSQL supports ALTER COLUMN
                print("Detected PostgreSQL database. Altering actual_class column...")
                db.session.execute(text("""
                    ALTER TABLE transactions 
                    ALTER COLUMN actual_class DROP NOT NULL
                """))
                
            else:
                print(f"Unsupported database engine: {engine_name}")
                return False
            
            db.session.commit()
            print("✅ Migration completed successfully!")
            
            # Verify the migration
            count_total = Transaction.query.count()
            count_labeled = Transaction.query.filter(Transaction.actual_class.isnot(None)).count()
            count_unlabeled = count_total - count_labeled
            
            print(f"📊 Migration summary:")
            print(f"   Total transactions: {count_total}")
            print(f"   Labeled transactions: {count_labeled}")
            print(f"   Unlabeled transactions: {count_unlabeled}")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            db.session.rollback()
            return False

def verify_migration():
    """Verify that the migration was successful"""
    with app.app_context():
        try:
            # Test inserting a transaction without actual_class
            test_transaction = Transaction()
            test_transaction.transaction_id = "migration_test"
            test_transaction.time_feature = 0.0
            test_transaction.amount = 100.0
            test_transaction.actual_class = None  # This should work now
            
            # Set all V features to 0
            for i in range(1, 29):
                setattr(test_transaction, f'v{i}', 0.0)
            
            db.session.add(test_transaction)
            db.session.commit()
            
            # Clean up test transaction
            db.session.delete(test_transaction)
            db.session.commit()
            
            print("✅ Migration verification successful!")
            return True
            
        except Exception as e:
            print(f"❌ Migration verification failed: {str(e)}")
            db.session.rollback()
            return False

if __name__ == "__main__":
    print("🔄 Starting database migration...")
    print("This will make the 'actual_class' column nullable to support unlabeled transactions.")
    
    # Confirm migration
    response = input("\nProceed with migration? (y/N): ").strip().lower()
    if response != 'y':
        print("Migration cancelled.")
        sys.exit(0)
    
    # Perform migration
    if migrate_actual_class_column():
        print("\n🔍 Verifying migration...")
        if verify_migration():
            print("\n🎉 Migration completed successfully!")
            print("\nYou can now upload CSV files with or without the Class column.")
            print("- With Class column: Transactions will be labeled for training")
            print("- Without Class column: Transactions will be unlabeled for prediction only")
        else:
            print("\n⚠️ Migration completed but verification failed.")
            print("Please check your database manually.")
    else:
        print("\n💥 Migration failed. Please check the error messages above.")
        sys.exit(1)
