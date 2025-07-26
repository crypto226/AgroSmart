import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import joblib
import os

# ---------- STEP 1: Load and Prepare Dataset ----------
def load_and_prepare_data(csv_path):
    try:
        df = pd.read_csv('Fertilizer_Prediction_gpt(1).csv')
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{csv_path}'")
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None, None, None, None, None, None, None

    # Define features and target
    numerical_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Temperature', 'Humidity']
    categorical_features = ['Soil Type', 'Crop Type']
    label = 'Fertilizer Name'

    # Ensure all required columns exist
    required_columns = numerical_features + categorical_features + [label]
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Error: Missing required column in dataset: '{col}'")
            return None, None, None, None, None, None, None, None

    X = df[numerical_features + categorical_features]
    y_raw = df[label]

    # Preprocessing: Scale numerical and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, "preprocessor.pkl")

    # Label encode the target variable
    fert_encoder = LabelEncoder()
    y_encoded = fert_encoder.fit_transform(y_raw)
    y_cat = to_categorical(y_encoded)
    joblib.dump(fert_encoder, "fertilizer_encoder.pkl")

    return X_processed, y_cat, X_processed.shape[1], y_cat.shape[1], preprocessor, fert_encoder, y_encoded, X

# ---------- STEP 2: Build and Train Hybrid Model ----------
def train_hybrid_model(X_train, y_train, input_dim, output_dim):
    # Reshape data for LSTM input (samples, time_steps, features)
    # Here time_steps is 1 as each sample is independent
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    model = Sequential([
        LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=False),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks for better training
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_fertilizer_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')

    model.fit(
        X_train_reshaped, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print("✅ Model trained and saved as 'best_fertilizer_model.h5'.")

# ---------- STEP 3: Recommend Fertilizer ----------
def recommend_fertilizer(input_data: dict, preprocessor, fert_encoder):
    # Load saved model
    model = load_model("best_fertilizer_model.h5")

    # Create DataFrame from new input
    input_df = pd.DataFrame([input_data])

    # Preprocess input using the saved preprocessor
    try:
        input_processed = preprocessor.transform(input_df)
    except ValueError as e:
        print(f"❌ Error during preprocessing: {e}")
        print("Please ensure your input data matches the format and types of the training data.")
        return

    # Reshape input for the LSTM model
    input_processed_reshaped = input_processed.reshape((input_processed.shape[0], 1, input_processed.shape[1]))


    # Predict
    prediction = model.predict(input_processed_reshaped)
    predicted_class = np.argmax(prediction)

    recommended_fertilizer = fert_encoder.inverse_transform([predicted_class])[0]
    print(f"\n🌱 Recommended Fertilizer: {recommended_fertilizer}")

# ---------- STEP 4: Run Everything ----------
if _name_ == "_main_":
    # === 1. Load your dataset ===
    dataset_path = input("📂 Enter path to your fertilizer CSV dataset: ").strip()
    if not os.path.exists(dataset_path):
        print("❌ Dataset file not found.")
        exit()

    X, y_cat, input_dim, output_dim, preprocessor, fert_encoder, y_encoded, X_original = load_and_prepare_data(dataset_path)

    if X is not None:  # Proceed only if data loaded successfully
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Train high-capacity hybrid model
        train_hybrid_model(X_train, y_train, input_dim, output_dim)

        # === 2. Get input from user ===
        print("\n🔢 Enter soil and crop parameters for fertilizer recommendation:")

        user_input = {}
        try:
            user_input['Nitrogen'] = float(input("Nitrogen (N): "))
            user_input['Phosphorus'] = float(input("Phosphorus (P): "))
            user_input['Potassium'] = float(input("Potassium (K): "))
            user_input['Moisture'] = float(input("Moisture (%): "))
            user_input['Temperature'] = float(input("Temperature (°C): "))
            user_input['Humidity'] = float(input("Humidity (%): "))
            user_input['Soil Type'] = input("Soil Type (e.g., Sandy, Loamy, Black): ").strip()
            user_input['Crop Type'] = input("Crop Type (e.g., Rice, Wheat, Maize): ").strip()

            recommend_fertilizer(user_input, preprocessor, fert_encoder)

        except ValueError:
            print("❌ Invalid input. Please enter correct numerical values.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")