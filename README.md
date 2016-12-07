# CptS483 project: using RCNN on ROS for object tracking


## Lab 4 Installation
Questions / notes for Lab 4
James will post the list of commands he used during his in-class tutorial here. Please ask more questions (or post tips) as they come up!
 
####To view ip address info:
ifconfig
 
####To setup your hosts file:
sudo gedit /etc/hosts
Then add a line similar to
 
192.168.1.127 irll-<name>
Change the ip address to be the ip address of the turtlebot, and set irll-<name> to be the hostname of the computer (irll-brian for example)
 
 
#### Setup your ~/.bashrc file to have the following lines:
export ROS_MASTER_URI=http://irll-brian:11311
export ROS_IP=$(ip addr | awk '/inet/ && /wlan0/{sub(/\/.*$/,"",$2); print $2}')
Make sure to replace irll-brian with the proper name for your turtlebot
The magic script for setting ROS_IP will just automatically grab your current wireless ip address.
 
 
#### on your computer, run

ssh turtlebot@irll-<name>
to connect to the turtlebot. Replace <name> with the name of your turtlebot
 
#### on the turtlebot, run:
roslaunch turtlebot_bringup minimal.launch
This will start the turtlebot drivers for movement
 
#### on the turtlebot, run:
roslaunch turtlebot_bringup 3dsensor.launch
This will start the turtlebot camera drivers
 
To drive the turtlebot with the keyboard, run (on either your computer or the turtlebot)
roslaunch turtlebot_teleop keyboard_teleop.launch
 
Troubleshooting:
roswtf
## CptS483: lab codes
https://github.com/IRLL/intro_to_robotics.git




#Questions / notes for Lab 4


 

##To view ip address info:

ifconfig

 

##To setup your hosts file:

sudo gedit /etc/hosts

Then add a line similar to

 

192.168.1.127 irll-<name>

Change the ip address to be the ip address of the turtlebot, and set irll-<name> to be the hostname of the computer (irll-brian for example)

 

 

## Setup your ~/.bashrc file to have the following lines:

export ROS_MASTER_URI=http://irll-brian:11311
export ROS_IP=$(ip addr | awk '/inet/ && /wlan0/{sub(/\/.*$/,"",$2); print $2}')

Make sure to replace irll-brian with the proper name for your turtlebot

The magic script for setting ROS_IP will just automatically grab your current wireless ip address.

 

 

##on your computer, run

ssh turtlebot@irll-<name>

to connect to the turtlebot. Replace <name> with the name of your turtlebot

 

on the turtlebot, run:

roslaunch turtlebot_bringup minimal.launch

This will start the turtlebot drivers for movement

 

##on the turtlebot, run:

roslaunch turtlebot_bringup 3dsensor.launch

This will start the turtlebot camera drivers

 

##To drive the turtlebot with the keyboard, run (on either your computer or the turtlebot)

roslaunch turtlebot_teleop keyboard_teleop.launch

 

##Troubleshooting:

roswtf
