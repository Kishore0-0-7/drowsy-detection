const int LED_PIN = 13;  // Use the built-in LED

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();

    if (receivedChar == '1') {
      // Face detected
      digitalWrite(LED_PIN, HIGH);
      Serial.println("Face detected");
    } else if (receivedChar == '0') {
      // No face detected
      digitalWrite(LED_PIN, LOW);
      Serial.println("No face detected");
    }

    // Clear any remaining characters in the buffer
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
}