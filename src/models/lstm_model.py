import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import joblib
import os

class LSTMModel:
    def __init__(self, input_shape=(10, 5), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid'))

        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy', 
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        if self.model is None:
            self.model = self.build_model()
        
        # Reshape data if needed
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], self.input_shape[0], self.input_shape[1])
        
        self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        self.is_trained = True

    def predict(self, X):
        if self.model is None or not self.is_trained:
            # Return dummy prediction for untrained model
            if len(X.shape) == 1:
                return np.array([0.1])  # Low fraud probability
            return np.array([[0.1]] * len(X))
        
        # Reshape data if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], self.input_shape[0], self.input_shape[1])
        elif len(X.shape) == 1:
            X = X.reshape(1, self.input_shape[0], self.input_shape[1])
        
        return self.model.predict(X)

    def save_model(self, filepath):
        if self.model:
            self.model.save(f"{filepath}.h5")
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")

    def load_model(self, filepath):
        from tensorflow.keras.models import load_model
        if os.path.exists(f"{filepath}.h5"):
            self.model = load_model(f"{filepath}.h5")
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            self.is_trained = True