#define in1 2
#define in2 3
#define in3 4
#define in4 5
#define enA 6
#define enB 9
#include <SoftwareSerial.h>
SoftwareSerial Bluetooth(10, 11); // RX, TX
//int LED = 13; // the on-board LED
int Data; // the data received
 
void setup() {
  Bluetooth.begin(9600);
  Serial.begin(9600);
  Serial.println("Waiting for command...");
  Bluetooth.println("Send 1 to turn on the LED. Send 0 to turn Off");
//  pinMode(LED,OUTPUT);

 
}
 
void loop()
{
  digitalWrite(enA, 50);
  digitalWrite(enB, 50);
  if (Bluetooth.available()){ //wait for data received
    Data=Bluetooth.read();
    
    if(Data=='1'){  
//      digitalWrite(LED,1);
//      Serial.println("LED On!");
//      Bluetooth.println("LED On!");
      digitalWrite(in1,HIGH); 
      digitalWrite(in2,LOW); 
      digitalWrite(in3,HIGH); 
      digitalWrite(in4,LOW);
    }
    else if(Data=='2'){
//     digitalWrite(LED,0);
//     Serial.println("LED Off!");
//     Bluetooth.println("LED  On D13 Off ! ");
     digitalWrite(in1,LOW); 
     digitalWrite(in2,HIGH); 
     digitalWrite(in3,LOW); 
     digitalWrite(in4,HIGH);
  }
    else if (Data=='3')
    {
    digitalWrite(in1,LOW); 
    digitalWrite(in2,LOW); 
    digitalWrite(in3,LOW); 
    digitalWrite(in4,LOW);
      }
  }
delay(100);
}
