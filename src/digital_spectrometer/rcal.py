from gpiozero import LED
import time

#setup the GPIO pins for the calibration states
gpio_pin0 = LED(21)
gpio_pin1 = LED(20)
gpio_pin2 = LED(16)
#gpio_pin3 = LED(12)

#each pin represents one bit iirc
#2^3 bits = 8 calibration states

def gpio_switch(n,t):
    global gpio_pin0, gpio_pin1, gpio_pin2
    if (n < 0 or n > 7) and n != 10:
        return "Invalid calibration index. Please choose a number between 0 and 7, or 10."
    
    idx = 7 - n
    #idx2 = 10 - n #for extra state

    #pin3 = idx2 & 8
    pin2 = idx & 4
    pin1 = idx & 2
    pin0 = idx & 1
    
    print(f"In switch state {n} with idx {idx}")

    if pin0:
        gpio_pin0.on()
    else:
        gpio_pin0.off()

    if pin1:
        gpio_pin1.on()
    else:
        gpio_pin1.off()

    if pin2:
        gpio_pin2.on()
    else:
        gpio_pin2.off()

    # if pin3:
    #     gpio_pin3.on()
    # else:
    #     gpio_pin3.off()

    # wait for 3 seconds just to allow the hw to catchup
    time.sleep(t)


def run_gpiozero(num,t):
	#num almost always starts at 0
	if num < 8:
		gpio_switch(num,t)
		return run_gpiozero(num + 1,t)
	elif num == 8:
		#return to initial state as currently the switching
		#occurs once after every saved sample
		gpio_switch(0,t)
		return 0
