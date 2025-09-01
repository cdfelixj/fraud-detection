# Database Streaming with Kafka

This feature allows you to stream existing transaction data from your PostgreSQL database through Kafka for real-time fraud detection processing, instead of using simulated data.

## How It Works

1. **On-Demand Control**: Use the "Start Database Streaming" button in the Kafka dashboard to begin streaming
2. **Real Database Data**: Streams actual transaction records from your `transactions` table
3. **Kafka Processing**: Data flows through Kafka topics and gets processed by the fraud detection consumer
4. **Real-time Results**: Fraud predictions appear in real-time on the dashboard

## Using Database Streaming

### Access the Kafka Dashboard
Visit: `http://localhost:5000/kafka/dashboard`

### Database Streaming Controls
- **Database Statistics**: Shows total records and fraud percentage in your database
- **Records Limit**: Optional limit on how many records to stream (leave empty for all)
- **Start Offset**: Skip this many records from the beginning (useful for continuing from where you left off)
- **Start Database Streaming**: Begin streaming data from the database to Kafka
- **Stop Streaming**: Stop the current streaming process
- **Refresh Stats**: Update the database statistics

### Configuration Options

**Stream a specific number of records:**
- Set "Records Limit" to 1000 to stream only 1000 records
- Leave empty to stream all records in the database

**Start from a specific position:**
- Set "Start Offset" to 500 to skip the first 500 records
- Useful for resuming streaming or testing different data segments

**Example Configurations:**
- Stream all records: Limit = empty, Offset = 0
- Stream 1000 records: Limit = 1000, Offset = 0  
- Stream 500 records starting from record 1000: Limit = 500, Offset = 1000

## Technical Details

### Data Flow
1. **DatabaseStreamer** reads transaction records from PostgreSQL
2. **Kafka Producer** sends transactions to the `fraud-detection-transactions` topic
3. **Kafka Consumer** processes transactions and generates predictions
4. **Results** are stored in the database and shown on the dashboard

### Streaming Rate
- Configurable batch size (default: 100 records per batch)
- Configurable interval between batches (default: 1 second)
- Can be adjusted via environment variables:
  - `DB_STREAM_BATCH_SIZE`: Records per batch
  - `DB_STREAM_INTERVAL`: Seconds between batches

### Database Requirements
Your database should have transaction records in the `transactions` table with the following columns:
- `id`: Primary key
- `time_feature`: Time feature for the transaction
- `v1` through `v28`: Feature columns
- `amount`: Transaction amount
- `actual_class`: Ground truth (0=normal, 1=fraud)
- `created_at`: Timestamp

## Benefits

1. **Real Data Testing**: Test your fraud detection system with real historical data
2. **Controlled Streaming**: Start and stop streaming on demand
3. **Batch Processing**: Process large datasets in manageable chunks
4. **Resume Capability**: Use offset to continue from where you stopped
5. **Performance Monitoring**: Monitor streaming rate and processing efficiency

## Monitoring

The Kafka dashboard provides real-time monitoring:
- **Streaming Status**: Shows if database streaming is active
- **Processing Rate**: Transactions processed per second
- **Live Feed**: Real-time log of streaming activities
- **Fraud Detection**: Live fraud alerts as they're detected

## Troubleshooting

**If streaming fails to start:**
- Check that Kafka is connected (green status indicator)
- Verify database connectivity
- Check the live feed for error messages

**If no data appears:**
- Verify your database has transaction records
- Check the database statistics in the dashboard
- Ensure the fraud detection models are trained

**Performance issues:**
- Reduce batch size via environment variables
- Increase interval between batches
- Monitor system resources during streaming

This feature gives you full control over when and how much data flows through your Kafka-based fraud detection system, perfect for testing, demonstrations, and controlled data processing scenarios.
