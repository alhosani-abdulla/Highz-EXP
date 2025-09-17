#include <SPI.h>
#include <LORegisterLibrary.h>

const int LEPen = 10;

int CEOn = 2;

int LOSetPin = 6;
int LOSetPinState = 1;
int prevLOSetPinState = 1;

int resetPin = 7;
int resetPinState = 1;
int prevResetPinState = 1;

int calibPin = 8;
int calibPinState = 1;
int prevCalibPinState = 1;
bool calib = false;

double freqOfLO = 650.0;    //current code structure increments immediately, so 650 is first freq
double stepSize = 2.0;
double calibStepSize = 0.2;

void writeADF4351Register(uint32_t reg) {
  digitalWrite(LEPen, LOW);
  SPI.transfer((reg >> 24) & 0xFF);
  SPI.transfer((reg >> 16) & 0xFF);
  SPI.transfer((reg >> 8) & 0xFF);
  SPI.transfer(reg & 0xFF);
  digitalWrite(LEPen, HIGH); // Latch data
  delayMicroseconds(10);
  digitalWrite(LEPen, LOW);
}

void setFreq(double freq) {
  
  int INT, MOD, FRAC, outputDiv;
  double bandSelClkDiv;
  calcRegs(freq, &INT, &MOD, &FRAC, &outputDiv, &bandSelClkDiv);

  uint32_t r0, r1, r2, r3, r4, r5;

  makeRegs(INT, FRAC, MOD, outputDiv, (int)bandSelClkDiv, &r0, &r1, &r2, &r3, &r4, &r5);
  //Serial.print("INT:");
  //Serial.println(INT);
  //Serial.print("FRAC:");
  //Serial.println(FRAC);
  //Serial.print("MOD:");
  //Serial.println(MOD);
  //Serial.print("OutputDiv:");
  //Serial.println(outputDiv);
  //Serial.print("clockDiv:");
  //Serial.println(bandSelClkDiv);

  writeADF4351Register(r5);
  writeADF4351Register(r4);
  writeADF4351Register(r3);
  writeADF4351Register(r2);
  writeADF4351Register(r1);
  writeADF4351Register(r0);
  Serial.print("Freq:");
  Serial.println(freq);
  
  //Serial.print("register 0: ");
  //Serial.println(r0);
  //Serial.print("register 1: ");
  //Serial.println(r1);
  //Serial.print("register 2: ");
  //Serial.println(r2);
  //Serial.print("register 3: ");
  //Serial.println(r3);
  //Serial.print("register 4: ");
  //Serial.println(r4);
  //Serial.print("register 5: ");
  //Serial.println(r5);
}

void setup() {
  pinMode(LEPen, OUTPUT);
  digitalWrite(LEPen, LOW);

  pinMode(CEOn, OUTPUT);
  digitalWrite(CEOn, HIGH);

  SPI.begin();
  SPI.beginTransaction(SPISettings(5000000, MSBFIRST, SPI_MODE0));
  Serial.begin(9600);

  pinMode(LOSetPin, INPUT);
  pinMode(resetPin, INPUT);
  pinMode(calibPin, INPUT);
}

bool incrementFreq(){
  LOSetPinState = digitalRead(LOSetPin);

  //4 cases below correspond to the 4 situations of pin state
  if (LOSetPinState == 0 and LOSetPinState != prevLOSetPinState){
    if (freqOfLO < 1000.0){                                            //protection to never set an LO over 9000MHz
      setFreq(freqOfLO);
    }
    prevLOSetPinState = 0;
  }

//other two conditions of maintaining state are not useful
  
  else if (LOSetPinState == 1 and LOSetPinState != prevLOSetPinState){
    
    if (calib == false and freqOfLO < 850.0 - stepSize){
      freqOfLO = freqOfLO + stepSize;
    }
    else if (calib == true and freqOfLO < 956.0 + 2.7/2 - calibStepSize){
      freqOfLO = freqOfLO + calibStepSize;
    }
    prevLOSetPinState = 1;
  }
}

void resetLO(){
  resetPinState = digitalRead(resetPin);
  //Serial.println(resetPinState);
  //Serial.println(prevResetPinState);
  //Serial.println(calib);

  if (resetPinState == 0 and prevResetPinState != resetPinState){
    if (calib == false) {
      freqOfLO = 650.0;
    }
    else if (calib == true) {
      freqOfLO = 904.0;
    }
    prevResetPinState = 0;
    Serial.println("Reseted the LO");
  }

  else if (resetPinState == 1 and prevResetPinState != resetPinState){
    prevResetPinState = 1;
  }
}

void checkCalib(){
  calibPinState = digitalRead(calibPin);
  //Serial.println(calib);

  //Serial.println("");
  //Serial.print("Current state:");
  //Serial.println(calibPinState);
  //Serial.print("Prev state:");
  //Serial.println(prevCalibPinState);
  //Serial.println("");

  if (calibPinState == 0 and prevCalibPinState != calibPinState){
    Serial.println("Making calib state True");
    calib = true;
    prevCalibPinState = 0;
  }

  else if (calibPinState == 1 and prevCalibPinState != calibPinState){
    Serial.println("Making calib state False");
    calib = false;
    prevCalibPinState = 1;
  }
  
}

void loop() {
  checkCalib();
  resetLO();
  incrementFreq();
}
