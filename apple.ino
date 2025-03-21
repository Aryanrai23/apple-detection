#include "DHT.h"

const int dhtPin = 27;           
const int npkAnalogPin = 34;     

#define DHTTYPE DHT11           
DHT dht(dhtPin, DHTTYPE);        // Initialize DHT sensor

void setup() {
  // Initialize Serial Communication
  Serial.begin(115200);

  // Start the DHT sensor
  dht.begin();

  Serial.println("DHT11 + NPK Sensor Data Transmission");
}

void loop() {
  // Read temperature and humidity from DHT11
  float temperature = dht.readTemperature(); 
  float humidity = dht.readHumidity();

  // Read analog value from NPK sensor
  int npkAnalogValue = analogRead(npkAnalogPin); 
  int npkPercent = map(npkAnalogValue, 0, 4095, 0, 100);

  // Calculate N, P, and K percentages
  float nitrogen = npkPercent * 0.40;   // 40% of NPK %
  float phosphorus = npkPercent * 0.30; // 30% of NPK %
  float potassium = npkPercent * 0.30;  // 30% of NPK %

  // Transmit data via Serial
  if (!isnan(temperature) && !isnan(humidity)) {
    Serial.print("Temperature: "); Serial.print(temperature); Serial.print("°C, ");
    Serial.print("Humidity: "); Serial.print(humidity); Serial.print("%, ");
    Serial.print("Nitrogen: "); Serial.print(nitrogen); Serial.print("%, ");
    Serial.print("Phosphorus: "); Serial.print(phosphorus); Serial.print("%, ");
    Serial.print("Potassium: "); Serial.print(potassium); Serial.println("%");
  } else {
    Serial.println("Failed to read from DHT sensor!");
  }

  // Delay for 5 seconds
  delay(5000);
}
