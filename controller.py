# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from pickle import FALSE

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt
import EasyGA

class controller(KesslerController):
    
    
        
    def __init__(self):
        self.eval_frames = 0 #What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python

        asteroid_time = ctrl.Antecedent(np.arange(0,10,0.1), 'asteroid_time') #how long till the nearest asteroid hits the ship
        asteroid_theta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'asteroid_theta') #Direction of asteroid relative to ship

        mine_distance = ctrl.Antecedent(np.arange(0,2000,10), 'mine_distance') #how close we are to a mine (range TBC)
        mine_theta = ctrl.Antecedent(np.arange(0,2000,10), 'mine_theta') #Direction mine relative to ship

        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        ship_thrust = ctrl.Consequent(np.arange(-1000,1000,10), 'ship_thrust')
        ship_mine = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_mine')
        ship_evade = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_evade') #if above 0 prioritize evading, if below prioritize shooting
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        #Declare fuzzy sets for asteroid_time (how long it takes for the nearest asteroid to reach the ship)
        asteroid_time['S'] = fuzz.trimf(asteroid_time.universe,[0,0,1])
        asteroid_time['M'] = fuzz.trimf(asteroid_time.universe, [0,1.5,3])
        asteroid_time['L'] = fuzz.smf(asteroid_time.universe,2,3)

        #Declare fuzzy sets for mine_distance (how long it takes for the nearest asteroid to reach the ship)
        mine_distance['S'] = fuzz.zmf(mine_distance.universe,150,175)
        mine_distance['M'] = fuzz.trimf(mine_distance.universe, [175,275,375])
        mine_distance['L'] = fuzz.smf(mine_distance.universe,350,400)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # NL - Negative Large, NM - Negative Small, etc.
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)

        # Declare fuzzy sets for asteroid_theta (degrees of turn needed to reach the calculated firing angle)
        # NL - Negative Large, NM - Negative Small, etc.
        # Hard-coded for a game step of 1/30 seconds
        asteroid_theta['NL'] = fuzz.zmf(asteroid_theta.universe, -1*math.pi/30,-2*math.pi/90)
        asteroid_theta['NM'] = fuzz.trimf(asteroid_theta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        asteroid_theta['NS'] = fuzz.trimf(asteroid_theta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # asteroid_theta['Z'] = fuzz.trimf(asteroid_theta.universe, [-1*math.pi/90,0,math.pi/90])
        asteroid_theta['PS'] = fuzz.trimf(asteroid_theta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        asteroid_theta['PM'] = fuzz.trimf(asteroid_theta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        asteroid_theta['PL'] = fuzz.smf(asteroid_theta.universe,2*math.pi/90,math.pi/30)

        # Declare fuzzy sets for mine_theta (degrees of turn needed to reach the calculated firing angle)
        # NL - Negative Large, NM - Negative Small, etc.
        # Hard-coded for a game step of 1/30 seconds
        mine_theta['NL'] = fuzz.zmf(mine_theta.universe, -1*math.pi/30,-2*math.pi/90)
        mine_theta['NM'] = fuzz.trimf(mine_theta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        mine_theta['NS'] = fuzz.trimf(mine_theta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # mine_theta['Z'] = fuzz.trimf(mine_theta.universe, [-1*math.pi/90,0,math.pi/90])
        mine_theta['PS'] = fuzz.trimf(mine_theta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        mine_theta['PM'] = fuzz.trimf(mine_theta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        mine_theta['PL'] = fuzz.smf(mine_theta.universe,2*math.pi/90,math.pi/30)
        
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])

        #Declare fuzzy set for ship movement
        #Fast-foward, Medium-Forward, etc.
        ship_thrust['FF'] = fuzz.trimf(ship_thrust.universe, [-1000,-1000,-600])
        ship_thrust['MF'] = fuzz.trimf(ship_thrust.universe, [-1000,-600,-300])
        ship_thrust['SF'] = fuzz.trimf(ship_thrust.universe, [-600,-300,100])
        ship_thrust['SB'] = fuzz.trimf(ship_thrust.universe, [-100,300,600])
        ship_thrust['MB'] = fuzz.trimf(ship_thrust.universe, [300,600,1000])
        ship_thrust['FB'] = fuzz.trimf(ship_thrust.universe, [600,1000,1000])

        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 

        #same idea as ship_fire above, just for mine and evading
        ship_mine['N'] = fuzz.trimf(ship_mine.universe, [-1,-1,0.0])
        ship_mine['Y'] = fuzz.trimf(ship_mine.universe, [0.0,1,1])    
        ship_evade['N'] = fuzz.trimf(ship_evade.universe, [-1,-1,0.0])
        ship_evade['Y'] = fuzz.trimf(ship_evade.universe, [0.0,1,1])    
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
     
        rule22 = ctrl.Rule(asteroid_time['M'] & asteroid_theta['NL'] & mine_distance['L'] & mine_theta['NL'], ship_evade['Y']) #placeholder rule so we can still run
        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()
     
     
        
        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        # self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        # self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        # self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)

        self.targeting_control.addrule(rule22)

    #calculates bullet time and angle needed
    def bullet_calc(self, ship_state: Dict, game_state: Dict):
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)

        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        return bullet_t, shooting_theta
        
    def asteroid_calc(self, ship_state: Dict, game_state: Dict):
        #stub function; will return the time till nearest asteroid hits, and the direction it's coming from
        return 0, 0
    
    def mine_calc(self, ship_state: Dict, game_state: Dict):
        #stub function; will return the nearest mine position, and the direction of it
        return 0, 0

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        #calculate bullet time and angle needed to shoot
        bullet_t, shooting_theta = self.bullet_calc(ship_state, game_state)
        asteroid_t, asteroid_theta = self.asteroid_calc(ship_state, game_state)
        mine_distance, mine_theta = self.mine_calc(ship_state,game_state)
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)

        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.input['asteroid_time'] = asteroid_t
        shooting.input['asteroid_theta'] = asteroid_theta
        shooting.input['mine_distance'] = mine_distance
        shooting.input['mine_theta'] = mine_theta
        
        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        thrust = -1000

        drop_mine = False
        
        self.eval_frames +=1
        
        #DEBUG
        print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Controller"