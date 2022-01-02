#include <AccelStepper.h>

#define zMotorPin1  8      // IN1 on the ULN2003 driver
#define zMotorPin2  9      // IN2 on the ULN2003 driver
#define zMotorPin3  10     // IN3 on the ULN2003 driver
#define zMotorPin4  11     // IN4 on the ULN2003 driver

#define tiltMotorPin1  4   // IN1 on the ULN2003 driver
#define tiltMotorPin2  5   // IN2 on the ULN2003 driver
#define tiltMotorPin3  6   // IN3 on the ULN2003 driver
#define tiltMotorPin4  7   // IN4 on the ULN2003 driver

#define stepsPerRevolution 4096

// Define the AccelStepper interface type; 4 wire motor in full step mode:
#define MotorInterfaceType 8 //4 wire, half step

int state = 0;
// Initialize with pin sequence IN1-IN3-IN2-IN4 for using the AccelStepper library with 28BYJ-48 stepper motor:
AccelStepper zStepper = AccelStepper(MotorInterfaceType, zMotorPin1, zMotorPin3, zMotorPin2, zMotorPin4);
AccelStepper tiltStepper = AccelStepper(MotorInterfaceType, tiltMotorPin1, tiltMotorPin3, tiltMotorPin2, tiltMotorPin4);

void setup() {
  // Set the maximum steps per second:
  zStepper.setMaxSpeed(1000);
  tiltStepper.setMaxSpeed(1000);
  Serial.begin(9600);
}

void runStepper(AccelStepper stepper, int dir, unsigned int steps, int stateBase)
{
  stepper.setCurrentPosition(0);
  stepper.setSpeed(dir * 1000);
  while (stepper.currentPosition() != dir * steps) {
    stepper.runSpeed();
    Serial.write(stateBase + (dir + 1) / 2);
  }
}

void move(int dir, unsigned int steps) {
  runStepper(zStepper, dir, steps, 2);
}

void tilt(int dir, unsigned int steps) {
  runStepper(tiltStepper, dir, steps, 4);
}

void loop() {

  int ib = 0;
  while (Serial.available() > 0) {
    ib = Serial.read();
  }
  
  if (ib == 2) {
    move(-1, 18 * stepsPerRevolution);
  }
  else if (ib == 3) {
    move(1, 18 * stepsPerRevolution);
  }
  else if (ib == 4) {
    tilt(-1, 2 * stepsPerRevolution);
  }
  else if (ib == 5) {
    tilt(1, 2 * stepsPerRevolution);
  }
  else if (ib == 6) {
    move(-1, 1 * stepsPerRevolution);
  }
  else if (ib == 7) {
    move(1, 1 * stepsPerRevolution);
  }
  else if (ib == 8) {
    tilt(-1, 1 * stepsPerRevolution);
  }
  else if (ib == 9) {
    tilt(1, 1 * stepsPerRevolution);
  }

  delay(10);
  Serial.write(1);
}
