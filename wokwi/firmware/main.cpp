/*
 * ESP32 + MPU6050 Wearable Controller Firmware (Phase 2 HITL Simulation)
 *
 * Transmits timestamped 6-axis IMU telemetry at 100 Hz over Serial (115200 baud).
 * Protocol format: IMU,timestamp_ms,ax,ay,az,gx,gy,gz
 * Units:
 *   ax, ay, az: m/s^2
 *   gx, gy, gz: rad/s
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// Pin Definitions
#define I2C_SDA 21
#define I2C_SCL 22
#define STATUS_LED 2
#define PAUSE_BTN 15

// Configuration
#define SAMPLE_RATE_HZ 100
#define SAMPLE_INTERVAL_MS (1000 / SAMPLE_RATE_HZ)

Adafruit_MPU6050 mpu;
bool mpu_initialized = false;
unsigned long last_sample_time = 0;
bool pause_state = false;
bool last_btn_state = HIGH;

// Optional Hardware Synthetic Test Mode (Fallback if physical I2C MPU6050 is omitted in simulation)
bool synthetic_test_mode = false;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000);

  pinMode(STATUS_LED, OUTPUT);
  pinMode(PAUSE_BTN, INPUT_PULLUP);
  digitalWrite(STATUS_LED, LOW);

  Wire.begin(I2C_SDA, I2C_SCL);

  // Initialize MPU6050
  if (mpu.begin(0x68, &Wire)) {
    mpu_initialized = true;
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    digitalWrite(STATUS_LED, HIGH);
    Serial.println("STATUS,MPU6050_OK");
  } else {
    mpu_initialized = false;
    Serial.println("ERROR,MPU6050_NOT_FOUND");
    Serial.println("STATUS,FALLBACK_SYNTHETIC_MODE_ENABLED");
    synthetic_test_mode = true;
  }
}

void send_imu_packet(unsigned long now, float ax, float ay, float az, float gx, float gy, float gz) {
  Serial.print("IMU,");
  Serial.print(now);
  Serial.print(",");
  Serial.print(ax, 4);
  Serial.print(",");
  Serial.print(ay, 4);
  Serial.print(",");
  Serial.print(az, 4);
  Serial.print(",");
  Serial.print(gx, 5);
  Serial.print(",");
  Serial.print(gy, 5);
  Serial.print(",");
  Serial.println(gz, 5);
}

void loop() {
  unsigned long now = millis();

  // Check Pause Button (Debounced toggle)
  bool btn_state = digitalRead(PAUSE_BTN);
  if (btn_state == LOW && last_btn_state == HIGH) {
    pause_state = !pause_state;
    if (pause_state) {
      Serial.println("STATUS,PAUSE_ENABLED");
      digitalWrite(STATUS_LED, LOW);
    } else {
      Serial.println("STATUS,PAUSE_DISABLED");
      digitalWrite(STATUS_LED, HIGH);
    }
    delay(50); // Debounce
  }
  last_btn_state = btn_state;

  // Process incoming serial command toggles
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "TEST_MODE_ON") {
      synthetic_test_mode = true;
      Serial.println("STATUS,SYNTHETIC_TEST_MODE_ON");
    } else if (cmd == "TEST_MODE_OFF" && mpu_initialized) {
      synthetic_test_mode = false;
      Serial.println("STATUS,SYNTHETIC_TEST_MODE_OFF");
    }
  }

  // 100 Hz Sampling Timer Loop
  if (now - last_sample_time >= SAMPLE_INTERVAL_MS) {
    last_sample_time = now;

    if (pause_state) {
      return; // Do not transmit motion data when paused
    }

    if (mpu_initialized && !synthetic_test_mode) {
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);

      // Adafruit MPU6050 returns accel in m/s^2 and gyro in rad/s directly
      send_imu_packet(now, a.acceleration.x, a.acceleration.y, a.acceleration.z,
                           g.gyro.x, g.gyro.y, g.gyro.z);
    } else {
      // Synthetic Motion Generator for Wokwi Testing Scenarios
      float t = now / 1000.0f;
      float ax = 0.05f * sin(2.0f * PI * 0.5f * t);
      float ay = 0.05f * cos(2.0f * PI * 0.5f * t);
      float az = 9.81f;

      // Synthetic yaw/pitch wrist rotation (rad/s)
      float gx = 0.01f * sin(2.0f * PI * 1.0f * t);
      float gy = 0.30f * sin(2.0f * PI * 0.2f * t);  // Pitch rotation
      float gz = 0.30f * cos(2.0f * PI * 0.2f * t);  // Yaw rotation

      send_imu_packet(now, ax, ay, az, gx, gy, gz);
    }
  }
}
