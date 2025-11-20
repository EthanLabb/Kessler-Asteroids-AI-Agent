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
        self.eval_frames = 0 # How many frames have been evaluated thus far (a counter). It doesnt really get used yet but we could use it for stats later if we want

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time') # Time (in seconds) that it will take a bullet to reach the intercept point
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python

        asteroid_time = ctrl.Antecedent(np.arange(0,10,0.1), 'asteroid_time') #how long till the nearest asteroid hits the ship
        asteroid_theta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'asteroid_theta') #Direction of asteroid relative to ship

        mine_distance = ctrl.Antecedent(np.arange(0,2000,10), 'mine_distance') #how close we are to a mine (range TBC)
        mine_theta = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.1),'mine_theta') # This should be in radians

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
        ship_thrust['Z'] = fuzz.trimf(ship_thrust.universe, [-100, 0, 100])

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

        # THRUST RULES:
        # If asteroid will hit very soon and it's roughly in front, thrust hard backward (evade strongly)
        rule_thrust1 = ctrl.Rule(asteroid_time['S'] & (asteroid_theta['NS'] | asteroid_theta['PS']),ship_thrust['FB'])
        # Slowly back away if asteroid coming towards us with medium time to impact 
        rule_thrust2 = ctrl.Rule(asteroid_time['M'] & (asteroid_theta['NS'] | asteroid_theta['PS']),ship_thrust['MB'])
        # Dont thrust hard when asteroid impact not imminent
        rule_thrust3 = ctrl.Rule(asteroid_time['L'],ship_thrust['Z'])

        # MINE RULES:
        # If a mine is very close, mark that we should drop a mine (and potentially evade)
        rule_mine1 = ctrl.Rule(mine_distance['S'],ship_mine['Y'])
        # If mines are far away, don't bother dropping one
        rule_mine2 = ctrl.Rule(mine_distance['L'],ship_mine['N'])

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
        self.targeting_control.addrule(rule_thrust1)
        self.targeting_control.addrule(rule_thrust2)
        self.targeting_control.addrule(rule_thrust3)
        self.targeting_control.addrule(rule_mine1)
        self.targeting_control.addrule(rule_mine2)




    # Helper functions
    def get_closest_asteroid(self, ship_state: Dict, game_state: Dict):
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
        return closest_asteroid




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
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        # Find the closest asteroid (disregards asteroid velocity)
        closest_asteroid = self.get_closest_asteroid(ship_state, game_state)

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        if closest_asteroid is None:
            # No asteroids, we should do nothing. Probably wont atually see this happen
            return 0.0, 0.0
        # Vector from asteriod to ship
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0] 
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]

        # Above vectors angle
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
        """
        Variables: - asteroid_time: time until a threatening asteroid reaches the ship (seconds)
                   - asteroid_theta: angle of that asteroid relative to ship heading (radians)

        We define the distance vector as asteroid -> ship, so:
            closing_speed_toward_ship > 0  => asteroid is moving toward the ship
            closing_speed_toward_ship <= 0 => moving away or sideways (not a threat)
        """
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        closest_in_time = None  # Will track the asteroid with the smallest time_to_impact

        for asteroid in game_state["asteroids"]:
            asteroid_pos_x = asteroid["position"][0]
            asteroid_pos_y = asteroid["position"][1]

            # Distance vector from asteroid to ship
            x_dist_asteroid_to_ship = ship_pos_x - asteroid_pos_x
            y_dist_asteroid_to_ship = ship_pos_y - asteroid_pos_y

            distance_to_ship = math.sqrt(x_dist_asteroid_to_ship**2 + y_dist_asteroid_to_ship**2)

            asteroid_vel_x = asteroid["velocity"][0]
            asteroid_vel_y = asteroid["velocity"][1]

            if distance_to_ship > 0:
                # Calculate the volocity of asteroid along the line defined as asteroid to ship... This is the dot product of (asteroid_vel_x, asteroid_vel_y) * (x_dist_asteroid_to_ship, y_dist_asteroid_to_ship)
                closing_speed_toward_ship = (asteroid_vel_x * x_dist_asteroid_to_ship + asteroid_vel_y * y_dist_asteroid_to_ship) / distance_to_ship
            else:
                closing_speed_toward_ship = 0.0

            if closing_speed_toward_ship <= 0: # Asteroid is not approaching
                continue

            time_to_impact = distance_to_ship / closing_speed_toward_ship

            if (closest_in_time is None) or (time_to_impact < closest_in_time["time_to_impact"]):
                closest_in_time = {
                    "asteroid": asteroid,
                    "time_to_impact": time_to_impact,
                    "x_dist_asteroid_to_ship": x_dist_asteroid_to_ship,
                    "y_dist_asteroid_to_ship": y_dist_asteroid_to_ship,
                }

        # No asteroid is currently moving toward us: return "safe" default values
        if closest_in_time is None:
            asteroid_time = 10.0   # large (safe) time
            asteroid_theta = 0.0   # neutral angle
            return asteroid_time, asteroid_theta

        asteroid_time = closest_in_time["time_to_impact"]

        # We stored asteroid->ship as (x_dist_asteroid_to_ship, y_dist_asteroid_to_ship).
        # For the angle, we want ship->asteroid, so flip the sign:
        x_ship_to_asteroid = -closest_in_time["x_dist_asteroid_to_ship"]
        y_ship_to_asteroid = -closest_in_time["y_dist_asteroid_to_ship"]

        angle_ship_to_asteroid = math.atan2(y_ship_to_asteroid, x_ship_to_asteroid)

        ship_heading_rad = ship_state["heading"] * math.pi / 180.0

        asteroid_theta = angle_ship_to_asteroid - ship_heading_rad

        asteroid_theta = (asteroid_theta + math.pi) % (2 * math.pi) - math.pi

        return asteroid_time, asteroid_theta


    
    def mine_calc(self, ship_state: Dict, game_state: Dict):
        #stub function; will return the nearest mine position, and the direction of it
        # Find the closest mine
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_mine = None
        
        for a in game_state["mines"]:
            #Loop through all mines, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_mine is None :
                # Does not yet exist, so initialize first mine as the minimum. Ugh, how to do?
                closest_mine = dict(mine = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_mine["dist"] > curr_dist:
                    # New minimum found
                    closest_mine["mine"] = a
                    closest_mine["dist"] = curr_dist

        if closest_mine is None:
            mine_distance = 10000 #return arbitrarily high distance if there is no mine (enough for fuzzy logic to not care)
            mine_theta = 0
        else:
            mine_distance = closest_mine["dist"]
            #calculate angle of mine to the ship
            mine_ship_x = ship_pos_x - closest_mine["mine"]["position"][0]
            mine_ship_y = ship_pos_y - closest_mine["mine"]["position"][1]
            mine_theta = math.atan2(mine_ship_y,mine_ship_x)

        # print(closest_mine)
        # print(mine_theta)

        return mine_distance, mine_theta

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
        
        

        # drop_mine = False
        shooting.compute()
        outputs = shooting.output

        turn_rate = float(outputs.get('ship_turn', 0.0))

        fire_value = outputs.get('ship_fire', -1.0)
        fire = bool(fire_value >= 0)

        raw_thrust = float(outputs.get('ship_thrust', 0.0))
        raw_thrust *= 0.3
        MAX_THRUST = 1000.0
        thrust = max(-MAX_THRUST, min(MAX_THRUST, raw_thrust))


        # If asteroid is approaching soon
        if asteroid_t < 2.0:
            # And it's mostly behind us (more than 90 degrees off the nose)
            if abs(asteroid_theta) > math.pi / 2:
                # We want to move forward in this case
                thrust = -abs(thrust) if thrust != 0 else -150.0

        # # If no asteroid is near barely move
        # if asteroid_t > 5.0:
        #     thrust *= 0.2

        mine_value = outputs.get('ship_mine', -1.0)
        drop_mine = bool(mine_value >= 0)
        
        evade_value = outputs.get('ship_evade', -1.0)
        evade = (evade_value >= 0)
        if evade and asteroid_t < 3.0:
            # Instead of turning to shooting_theta, turn away from the asteroid.
            # asteroid_theta is the angle from ship heading to asteroid:
            #   > 0 means asteroid is to the left, < 0 means to the right.
            # To turn AWAY, we rotate in the opposite direction.
            if asteroid_theta > 0:
                # asteroid on left -> turn right
                turn_rate = -90.0
            else:
                # asteroid on right -> turn left
                turn_rate = 90.0

            # Also, don't fire while in pure evade mode
            fire = False



        #DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        # THe games thrust sign is opposite of our convention so keep the - in front of thrust 
        return -thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Controller"