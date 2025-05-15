import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from scipy.signal import butter, lfilter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def simulate_sensor_data():
    acc = np.random.normal(0, 0.1, 100)
    strain = np.random.normal(0.02, 0.005, 100)
    temp = np.random.normal(25, 1, 100)
    return acc, strain, temp

def butter_lowpass_filter(data, cutoff=0.5, fs=10.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def build_model(input_shape):
    model = Sequential([
        Conv1D(16, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def send_alert(message):
    sender = 'your_email@example.com'
    receiver = 'receiver_email@example.com'
    msg = MIMEText(message)
    msg['Subject'] = 'Structural Health Alert'
    msg['From'] = sender
    msg['To'] = receiver
    with smtplib.SMTP_SSL('smtp.example.com', 465) as server:
        server.login(sender, 'your_password')  # Replace with your actual password or use environment variables
        server.send_message(msg)

def monitor_structure(model):
    for i in range(10):
        acc, strain, temp = simulate_sensor_data()
        acc_filtered = butter_lowpass_filter(acc)
        data = np.array(acc_filtered).reshape((1, 100, 1))
        prediction = model.predict(data, verbose=0)
        if prediction[0][0] > 0.5:
            print(f"[ALERT] Possible structural anomaly detected! Prediction: {prediction[0][0]:.2f}")
            send_alert("Structural anomaly detected at iteration " + str(i))
        else:
            print(f"[OK] Structure is healthy. Prediction: {prediction[0][0]:.2f}")
        time.sleep(1)
        
if __name__ == "__main__":
    dummy_model = build_model((100, 1))
    # Dummy training for simulation
    dummy_model.fit(np.random.rand(10, 100, 1), np.random.randint(0, 2, 10), epochs=1, verbose=0)
    print("Starting Structural Health Monitoring System...")
    monitor_structure(dummy_model)
