# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from pickle import FALSE

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import Scenario, KesslerController, KesslerGame, GraphicsType, TrainerEnvironment # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt
import EasyGA
import random
import time
NUM_GENES = 12
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0
game_num = [1]

def three_sorted_points(min_val, max_val, g0, g1, g2):
        pts = [
            min_val + g0 * (max_val - min_val),
            min_val + g1 * (max_val - min_val),
            min_val + g2 * (max_val - min_val),
        ]
        pts.sort()
        return pts[0], pts[1], pts[2]
class controller(KesslerController):
        
    def __init__(self, mf_chromosome=None):
        self.eval_frames = 0 # How many frames have been evaluated thus far (a counter). It doesnt really get used yet but we could use it for stats later if we want
        self.framecounter = 15
        self.mf_chromosome = mf_chromosome

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time') # Time (in seconds) that it will take a bullet to reach the intercept point
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python

        asteroid_time = ctrl.Antecedent(np.arange(0,10,0.1), 'asteroid_time') #how long till the nearest asteroid hits the ship
        asteroid_theta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'asteroid_theta') #Direction of asteroid relative to ship

        mine_distance = ctrl.Antecedent(np.arange(0,2000,10), 'mine_distance') #how close we are to a mine (range TBC)
        mine_theta = ctrl.Antecedent(np.arange(-math.pi/30, math.pi/30, 0.1),'mine_theta') # This should be in radians

        currentrisk = ctrl.Antecedent(np.arange(0,50,0.1), 'currentrisk') #if our current space is safe (rating from 0-20)
        currentmine = ctrl.Antecedent(np.arange(0,30,1), 'currentmine') #how many asteroids are going to hit where we are
        bestdirection = ctrl.Antecedent(np.arange(-math.pi/30,math.pi/30, 0.1), 'bestdirection') #direction with safest rating
        shootasteroid = ctrl.Antecedent(np.arange(-1,1,0.1), 'shootasteroid') #if we will hit an asteroid (boolean)

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
        mine_distance['M'] = fuzz.trimf(mine_distance.universe, [0,400,800])
        mine_distance['L'] = fuzz.smf(mine_distance.universe,600,2000)

        #Declare fuzzy sets for currentmine (mine rating via how many asteroids are incoming)
        currentmine['S'] = fuzz.trimf(currentmine.universe,[0,0,4])
        currentmine['M'] = fuzz.trimf(currentmine.universe, [0,8,15])
        currentmine['L'] = fuzz.smf(currentmine.universe,10,20)

        #Declare fuzzy sets for currentmine (mine rating via how many asteroids are incoming)
        currentrisk['S'] = fuzz.trimf(currentrisk.universe,[0,0,30])
        currentrisk['L'] = fuzz.trimf(currentrisk.universe,[25,50,50])
        
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

        # Declare fuzzy sets for bestdirection (degrees of turn needed to reach the calculated firing angle)
        # NL - Negative Large, NM - Negative Small, etc.
        # Hard-coded for a game step of 1/30 seconds
        bestdirection['NL'] = fuzz.zmf(bestdirection.universe, -1*math.pi/30,-2*math.pi/90)
        bestdirection['NM'] = fuzz.trimf(bestdirection.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        bestdirection['NS'] = fuzz.trimf(bestdirection.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        bestdirection['Z'] = fuzz.gaussmf(bestdirection.universe, 0, math.pi/120)
        bestdirection['PS'] = fuzz.trimf(bestdirection.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        bestdirection['PM'] = fuzz.trimf(bestdirection.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        bestdirection['PL'] = fuzz.smf(bestdirection.universe,2*math.pi/90,math.pi/30)

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

        #other boolean fuzzys, exact same idea as above
        ship_mine['N'] = fuzz.trimf(ship_mine.universe, [-1,-1,0.0])
        ship_mine['Y'] = fuzz.trimf(ship_mine.universe, [0.0,1,1])    
        ship_evade['N'] = fuzz.trimf(ship_evade.universe, [-1,-1,0.0])
        ship_evade['Y'] = fuzz.trimf(ship_evade.universe, [0.0,1,1])    
        shootasteroid['N'] = fuzz.trimf(shootasteroid.universe, [-1,-1,0.0])
        shootasteroid['Y'] = fuzz.trimf(shootasteroid.universe, [0.0,1,1])    
        

        if self.mf_chromosome is not None:
            
            g0, g1, g2 = mf_chromosome[0], mf_chromosome[1], mf_chromosome[2]
            t1, t2, t3 = three_sorted_points(0.2, 5.0, g0, g1, g2)

            asteroid_time['S'] = fuzz.trimf(asteroid_time.universe, [0.0, 0.0, t1])
            asteroid_time['M'] = fuzz.trimf(asteroid_time.universe, [t1, t2, t3])
            asteroid_time['L'] = fuzz.smf(asteroid_time.universe, t2, 10.0)


            g3, g4, g5 = mf_chromosome[3], mf_chromosome[4], mf_chromosome[5]
            r1, r2, r3 = three_sorted_points(0.0, 50.0, g3, g4, g5)

            # Safe up to around r2
            currentrisk['S'] = fuzz.trimf(currentrisk.universe, [0.0, 0.0, r2])

            # High risk from around r1 upwards
            currentrisk['L'] = fuzz.trimf(currentrisk.universe, [r1, 50.0, 50.0])
            


            g6, g7, g8 = mf_chromosome[6], mf_chromosome[7], mf_chromosome[8]
            d1, d2, d3 = three_sorted_points(100.0, 1200.0, g6, g7, g8)

            mine_distance['S'] = fuzz.zmf(mine_distance.universe, d1, d2)
            mine_distance['M'] = fuzz.trimf(mine_distance.universe, [0.0, d2, d3])
            mine_distance['L'] = fuzz.smf(mine_distance.universe, d3, 2000.0)


            g9, g10, g11 = mf_chromosome[9], mf_chromosome[10], mf_chromosome[11]

            # Zero band half-width in [50, 250]
            zero_half_width = 50.0 + g9 * 200.0

            # Backward (positive) thrust scale [600, 1000]
            backward_max = 600.0 + g10 * 400.0

            # Forward (negative) thrust scale [600, 1000]
            forward_max = 600.0 + g11 * 400.0

            # Zero (small thrust)
            ship_thrust['Z']  = fuzz.trimf(ship_thrust.universe,
                                        [-zero_half_width, 0.0, zero_half_width])

            # Forward thrust (negative values)
            ship_thrust['FF'] = fuzz.trimf(ship_thrust.universe,
                                        [-1000.0, -1000.0, -forward_max])
            ship_thrust['MF'] = fuzz.trimf(ship_thrust.universe,
                                        [-1000.0, -forward_max, -0.5 * forward_max])
            ship_thrust['SF'] = fuzz.trimf(ship_thrust.universe,
                                        [-forward_max, -0.5 * forward_max, 0.0])

            # Backward thrust (positive values)
            ship_thrust['SB'] = fuzz.trimf(ship_thrust.universe,
                                        [0.0, 0.5 * backward_max, backward_max])
            ship_thrust['MB'] = fuzz.trimf(ship_thrust.universe,
                                        [0.5 * backward_max, backward_max, backward_max])
            ship_thrust['FB'] = fuzz.trimf(ship_thrust.universe,
                                        [backward_max, 1000.0, 1000.0])




        #Declare each fuzzy rule
        #targeting algorithm rules
        rule_target1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_mine['N']))
        rule_target2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_mine['N']))
        rule_target3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_mine['N']))
        # rule_target4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule_target5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_mine['N']))
        rule_target6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_mine['N']))
        rule_target7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_mine['N']))
        rule_target8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_mine['N']))
        rule_target9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_mine['N']))
        rule_target10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_mine['N']))
        # rule_target11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule_target12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_mine['N']))
        rule_target13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_mine['N']))
        rule_target14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_mine['N']))
        rule_target15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_mine['N']))
        rule_target16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_mine['N']))
        rule_target17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_mine['N']))
        # rule_target18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule_target19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_mine['N']))
        rule_target20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_mine['N']))
        rule_target21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_mine['N']))
     
        #evade movement rules/thrust rules
        rule_move1 = ctrl.Rule(currentrisk['L'] & bestdirection['PL'], (ship_turn['NS'], ship_thrust['FB'], ship_mine['Y']))
        rule_move2 = ctrl.Rule(currentrisk['S'] & bestdirection['PL'], (ship_turn['NS'], ship_thrust['FB'], ship_mine['N']))
        rule_move3 = ctrl.Rule(bestdirection['PM'], (ship_turn['NL'], ship_thrust['Z'], ship_mine['N']))
        rule_move4 = ctrl.Rule(bestdirection['PS'], (ship_turn['PS'], ship_thrust['FF'], ship_mine['N']))
        rule_move5 = ctrl.Rule(bestdirection['Z'], (ship_thrust['FF'], ship_mine['Y']))
        rule_move6 = ctrl.Rule(bestdirection['NS'], (ship_turn['NS'], ship_thrust['FF'], ship_mine['N']))
        rule_move7 = ctrl.Rule(bestdirection['NM'], (ship_turn['NL'], ship_thrust['Z'], ship_mine['N']))
        rule_move8 = ctrl.Rule(currentrisk['S'] & bestdirection['NL'], (ship_turn['PS'], ship_thrust['FB'], ship_mine['N']))
        rule_move9 = ctrl.Rule(currentrisk['L'] & bestdirection['NL'], (ship_turn['PS'], ship_thrust['FB'], ship_mine['Y']))
        rule_move10 = ctrl.Rule(currentrisk['S'], (ship_thrust['Z'], ship_mine['N']))

        #evade status rules:
        rule_evade1 = ctrl.Rule(currentrisk['L'] | mine_distance['S'] | asteroid_time['S'], ship_evade['Y'])
        rule_evade2 = ctrl.Rule(currentrisk['S'] & mine_distance['L'] & asteroid_time['M'], ship_evade['N'])
        rule_evade3 = ctrl.Rule(currentrisk['S'] & mine_distance['L'] & asteroid_time['L'], ship_evade['N'])
        rule_evade4 = ctrl.Rule(currentrisk['S'] & mine_distance['M'] & asteroid_time['M'], ship_evade['N'])
        rule_evade5 = ctrl.Rule(currentrisk['S'] & mine_distance['M'] & asteroid_time['L'], ship_evade['N'])

        #fire rules:
        rule_fire1 = ctrl.Rule(shootasteroid['Y'], ship_fire['Y'])
        rule_fire2 = ctrl.Rule(shootasteroid['N'], ship_fire['N'])

        # THRUST RULES:
        # If asteroid will hit very soon and it's roughly in front, thrust hard backward (evade strongly)
        #rule_thrust1 = ctrl.Rule(asteroid_time['S'] & (asteroid_theta['NS'] | asteroid_theta['PS']),ship_thrust['FB'])
        # Slowly back away if asteroid coming towards us with medium time to impact 
        #rule_thrust2 = ctrl.Rule(asteroid_time['M'] & (asteroid_theta['NS'] | asteroid_theta['PS']),ship_thrust['MB'])
        # Dont thrust hard when asteroid impact not imminent
        #rule_thrust3 = ctrl.Rule(asteroid_time['L'],ship_thrust['Z'])

        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()
     
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule_target1)
        self.targeting_control.addrule(rule_target2)
        self.targeting_control.addrule(rule_target3)
        # self.targeting_control.addrule(rule_target4)
        self.targeting_control.addrule(rule_target5)
        self.targeting_control.addrule(rule_target6)
        self.targeting_control.addrule(rule_target7)
        self.targeting_control.addrule(rule_target8)
        self.targeting_control.addrule(rule_target9)
        self.targeting_control.addrule(rule_target10)
        # self.targeting_control.addrule(rule_target11)
        self.targeting_control.addrule(rule_target12)
        self.targeting_control.addrule(rule_target13)
        self.targeting_control.addrule(rule_target14)
        self.targeting_control.addrule(rule_target15)
        self.targeting_control.addrule(rule_target16)
        self.targeting_control.addrule(rule_target17)
        # self.targeting_control.addrule(rule_target18)
        self.targeting_control.addrule(rule_target19)
        self.targeting_control.addrule(rule_target20)
        self.targeting_control.addrule(rule_target21)

        self.movement_control = ctrl.ControlSystem()
        self.movement_control.addrule(rule_move1)
        self.movement_control.addrule(rule_move2)
        self.movement_control.addrule(rule_move3)
        self.movement_control.addrule(rule_move4)
        self.movement_control.addrule(rule_move5)
        self.movement_control.addrule(rule_move6)
        self.movement_control.addrule(rule_move7)
        self.movement_control.addrule(rule_move8)
        self.movement_control.addrule(rule_move9)
        self.movement_control.addrule(rule_move10)



        #self.targeting_control.addrule(rule_thrust1)
        #self.targeting_control.addrule(rule_thrust2)
        #self.targeting_control.addrule(rule_thrust3)
        #self.targeting_control.addrule(rule_mine1)
        #self.targeting_control.addrule(rule_mine2)

        self.evade_control = ctrl.ControlSystem()
        self.evade_control.addrule(rule_evade1)
        self.evade_control.addrule(rule_evade2)
        self.evade_control.addrule(rule_evade3)
        self.evade_control.addrule(rule_evade4)
        self.evade_control.addrule(rule_evade5)

        self.fire_control = ctrl.ControlSystem()
        self.fire_control.addrule(rule_fire1)
        self.fire_control.addrule(rule_fire2)



    
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

    #calculates if a bullet fired will hit an asteroid 
    #most of this code is copied from bulletcalc/closest asteroid
    def asteroid_hit_calc(self, ship_state: Dict, game_state: Dict):
        shootasteroid = -1 #our output, default no asteroid in sights
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        for a in game_state['asteroids']:
            asteroid_ship_x = ship_pos_x - a["position"][0] 
            asteroid_ship_y = ship_pos_y - a["position"][1]
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)

            # Above vectors angle
            asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
            
            asteroid_direction = math.atan2(a["velocity"][1], a["velocity"][0]) # Velocity is a 2-element array [vx,vy].
            my_theta2 = asteroid_ship_theta - asteroid_direction 
            cos_my_theta2 = math.cos(my_theta2)
            # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
            asteroid_vel = math.sqrt(a["velocity"][0]**2 + a["velocity"][1]**2)
            bullet_speed = 800 # Hard-coded bullet speed from bullet.py
            
            # Determinant of the quadratic formula b^2-4ac
            targ_det = (-2 * curr_dist * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (curr_dist**2))
            
            # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
            intrcpt1 = ((2 * curr_dist * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
            intrcpt2 = ((2 * curr_dist * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
            
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
            intrcpt_x = a["position"][0] + a["velocity"][0] * (bullet_t+1/30)
            intrcpt_y = a["position"][1] + a["velocity"][1] * (bullet_t+1/30)

            
            my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
            
            # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
            shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
            
            # Wrap all angles to (-pi, pi)
            shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

            #test if we will currently hit the asteroid
            if abs(shooting_theta) < 0.1:
                shootasteroid = 1
                break

        return shootasteroid

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


    def rect_calc(self, ship_state: Dict, game_state: Dict):
        #This function divides the gamespace into a grid of squares, and fuzzily rate those square on how safe they are
        #By using this function the ship should be able to determine if it should move or not. 
        #If there are no safe squares nearby it should probably try to shoot an asteroid to escape the trap

        #These rects should be larger than the ship itself
        #For this we define 10x10 grid
        gridsize = 10
        mapsizex = game_state['map_size'][0]
        mapsizey = game_state['map_size'][1] 
        rectsizex = mapsizex/gridsize
        rectsizey = mapsizey/gridsize

        #generate square grid
        grid = [[{"mineexists": False, "asteroids_incoming": 0, "asteroids_time":  1000, "nearest_asteroid":  1000}for i in range(gridsize)] for j in range(gridsize)] 
        safetygrid = [[0 for i in range(gridsize)] for j in range(gridsize)] #this one will contain our actual ratings from fuzzy logic
        minegrid = [[0 for i in range(gridsize)] for j in range(gridsize)]  #will contain a mine score (i.e. if a lot of asteroids are going to pass through this point)

        #return values
        currentrisk = 0 #weighted safety score; see below
        currentmine = 0 #just returning the amount of asteroids incoming
        bestdirection = 0 #will be a radian

        #Loop through every point in the grid
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                #define the bounds for our square
                x_min = rectsizex * (i)
                x_max = rectsizex * (i+1)
                y_min = rectsizey * (j)
                y_max = rectsizey * (j+1)
                
                #===logic for mines===
                #tests if a mine exists in this area
                for a in game_state["mines"]:
                    if (a["position"][0] > x_min and a["position"][0] < x_max and a["position"][1] > y_min and a["position"][1] < y_max):
                        grid[i][j]["mineexists"] = True
                        break
            
                #===logic for asteroids===
                for asteroid in game_state["asteroids"]:
                    #asteroid["position"][0] #x pos
                    #asteroid["position"][1] #y pos
                    #asteroid["velocity"][0] #x vel
                    #asteroid["velocity"][1] #y vel

                    #calculate direction of asteroid velocity
                    asteroid_direction = math.atan2(asteroid["velocity"][1], asteroid["velocity"][0])

                    #find the angles to the different corners of the square based on the asteroid position
                    angle1 = math.atan2(y_min - asteroid["position"][1], x_min - asteroid["position"][0])
                    angle2 = math.atan2(y_max - asteroid["position"][1], x_min - asteroid["position"][0])
                    angle3 = math.atan2(y_min - asteroid["position"][1], x_max - asteroid["position"][0])
                    angle4 = math.atan2(y_max - asteroid["position"][1], x_max - asteroid["position"][0])

                    #find the max and min of these angles
                    anglemax = max(angle1, angle2, angle3, angle4)
                    anglemin = min(angle1, angle2, angle3, angle4)

                    #if the asteroid direction isn't in this angle range, it's not going to intersect and we move onto the next asteroid
                    if((asteroid_direction > anglemax) or (asteroid_direction < anglemin)):
                        continue
                    
                    #if we get here the asteroid is on collision path, update grid variable accordingly
                    grid[i][j]["asteroids_incoming"] = grid[i][j]["asteroids_incoming"] + 1


                    #calculate magnitude of asteroid velocity
                    asteroid_speed = abs(sqrt(asteroid["velocity"][1]**2 + asteroid["velocity"][0]**2))

                    #calculate asteroid distance to center of square
                    asteroid_dist = abs(sqrt((x_min+rectsizex/2-asteroid["position"][0])**2+(y_min+rectsizey/2-asteroid["position"][1])**2))

                    #calculate intersect time
                    #The game speed is 30 ticks per second, scale accordingly
                    intersect_time = asteroid_dist/asteroid_speed*30
                    
                    #update to lowest
                    if grid[i][j]["asteroids_time"] > intersect_time:
                        grid[i][j]["asteroids_time"] = intersect_time
                    
                    if grid[i][j]["nearest_asteroid"] > asteroid_dist:
                        grid[i][j]["nearest_asteroid"] = asteroid_dist

                        
                    

                #calculate safety grid values
                #higher rating is worse
                
                #to add to neighbours as well; this method isn't precise, so make nearby squares also dangerous
                neighbourfactor = 0
                #neighbourfactor += (grid[i][j]["asteroids_time"]**-1)*5
                neighbourfactor += grid[i][j]["asteroids_incoming"]

                if grid[i][j]["mineexists"] == True:
                    #obviously if there is a mine the safe isn't safe, but it should also set nearby spaces to be unsafe aswell
                    safetygrid[i][j] += 1000
                    neighbourfactor += 1000

                for n in range(3):
                    for m in range(3):
                        #temporary variables for wrapping
                        x = i
                        y = j

                        #positive wrapping
                        try:
                            safetygrid[i+n]
                            x = i
                        except: 
                            x = 0
                                
                        try:
                            safetygrid[j+n]
                            y = j
                        except: 
                            y = 0                               
                            safetygrid[x+n][y+m] += neighbourfactor

                        #negative wrapping
                        try:
                            safetygrid[i-n]
                            x = i
                        except: 
                            x = len(grid)
                                
                        try:
                            safetygrid[j-n]
                            y = j
                        except: 
                            y = len(grid[0])     

                        safetygrid[i-n][j-m] += neighbourfactor

                #other rating input:
                safetygrid[i][j] += (grid[i][j]["asteroids_time"]**-1)*5
                safetygrid[i][j] += grid[i][j]["asteroids_incoming"]
                safetygrid[i][j] += (grid[i][j]["nearest_asteroid"]**-1)*250
                minegrid[i][j] = grid[i][j]["asteroids_incoming"]

                #make the borders more scary since we aren't calculating asteroid math considering wrapping right now
                if (i == 0) or (i == len(grid) - 1) or (j == 0) or (j == len(grid[i]) - 1):
                    safetygrid[i][j] += 20
        
        #find the mine status of the ship, there's probably a more efficient way of doing this but this works for now
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]    
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                x_min = rectsizex * (i)
                x_max = rectsizex * (i+1)
                y_min = rectsizey * (j)
                y_max = rectsizey * (j+1)

                if ship_pos_x <= x_max and ship_pos_x >= x_min and ship_pos_y <= y_max and ship_pos_y >= y_min:
                    #current ship square, find related results
                    currentmine = minegrid[i][j]
                    #redefine these values for next function
                    ship_pos_x = i
                    ship_pos_y = j
                    break
        
        #cap currentmine value for fuzzy logic purposes
        if currentmine > 30:
            currentmine = 30

        #evaluate 8 directions and see what's safe
        """
        4 represents ship below, other numbers are array indicies of direction ratings
        0 1 2
        3 4 5
        6 7 8

        The direction rating takes a weighted average in each direction, hoping by weighting the values this way the ship won't cross impassible areas 
        The formula follows fibonacci sequence to weight near values higher, i.e.:
        Left rating = (8(x-1, y) + 5(x-2, y), 3(x-3, y), 2(x-4, y), 1(x-5, y))/(8+5+3+2+1)

        """
        directionratings = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        distanceeval = 5 #how many squares out we evaluate in this function
        sumval = 1 #for weighting
        lastval = 0 #for weighting
        lastval2 = 0 #for weighting
        avgtotal =  0 #for division at end

        #variables for wrapping
        #negatives start at 1 since last index is [gridsize - 1], while positives first index is [0]
        negxcount = 1
        posxcount = 0
        negycount = 1
        posycount = 0


        for j in range(distanceeval+1):
            if j == 0: continue #we don't care about this case
            avgtotal += (sumval + lastval)
            i = distanceeval + 1 - j #going backwards to make math easier

            #catch errors for wrapping
            try: 
                safetygrid[ship_pos_x - i]
                negx = ship_pos_x - i
            except:
                negx = gridsize - negxcount
                negxcount += 1

            try: 
                safetygrid[ship_pos_x + i]
                posx = ship_pos_x + i
            except: 
                posx = posxcount
                posxcount += 1

            try: 
                safetygrid[ship_pos_y - i]
                negy = ship_pos_y - i
            except: 
                negy = gridsize - negycount
                negycount += 1

            try: 
                safetygrid[ship_pos_y + i]
                posy = ship_pos_y + i
            except: 
                posy = posycount
                posycount += 1
            
            #update direction ratings
            directionratings[0] += (sumval + lastval)*safetygrid[negx][posy]
            directionratings[1] += (sumval + lastval)*safetygrid[ship_pos_x][posy]
            directionratings[2] += (sumval + lastval)*safetygrid[posx][posy]
            directionratings[3] += (sumval + lastval)*safetygrid[negx][ship_pos_y]
            directionratings[5] += (sumval + lastval)*safetygrid[posx][ship_pos_y]
            directionratings[6] += (sumval + lastval)*safetygrid[negx][negy]
            directionratings[7] += (sumval + lastval)*safetygrid[ship_pos_x][negy]
            directionratings[8] += (sumval + lastval)*safetygrid[posx][negy]

            #update sequence
            lastval2 = sumval
            sumval = sumval + lastval 
            lastval = lastval2

        #DIRECTION RATINGS DEBUG
        #print("Directions:")
        #print(directionratings[0], directionratings[1], directionratings[2])
        #print(directionratings[3], directionratings[4], directionratings[5])
        #print(directionratings[6], directionratings[7], directionratings[8])    

        #find the best direction
        bestdirection = 4
        directionratings[4] = safetygrid[ship_pos_x][ship_pos_y]
        #currentsafety = directionratings[4] #default assumption that our current position (dir 4) is the best option

        for i in range(len(directionratings)):
            if i == 4: continue #skip mid point
            directionratings[i] /= avgtotal #normalize
            if directionratings[i] < directionratings[bestdirection]:
                bestdirection = i
                #currentsafety = -1 # our current position (4) is not the best option

        currentrisk = directionratings[4]*50/max(directionratings) #assign our current risk

        #convert best direction to radians
        match bestdirection:
            case 0:
                bestdirection = 3*math.pi/4

            case 1: 
                bestdirection = math.pi/2

            case 2:
                bestdirection = math.pi/4

            case 3:
                bestdirection = math.pi
            
            case 4:
                #center case, return 0
                bestdirection = 0

            case 5:
                bestdirection = 0

            case 6:
                bestdirection = -3*math.pi/4

            case 7:
                bestdirection = -math.pi/2

            case 8:
                bestdirection = -math.pi/4
        
        #redefine best direction with respect to ships direction
        bestdirection = bestdirection - ((math.pi/180)*ship_state["heading"])

        #redefine domain from -pi to pi
        if bestdirection > math.pi:
            bestdirection -= 2*math.pi
        elif bestdirection < -1*math.pi:
            bestdirection += 2*math.pi
        

        return currentrisk, currentmine, bestdirection

    
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
        currentrisk, currentmine, bestdirection = self.rect_calc(ship_state,game_state)
        shootasteroid = self.asteroid_hit_calc(ship_state, game_state)
        
        # Pass the inputs to the rulebase and fire it
        evading = ctrl.ControlSystemSimulation(self.evade_control,flush_after_run=1)
        firesim = ctrl.ControlSystemSimulation(self.fire_control,flush_after_run=1)

        # Get evade rating
        evading.input['currentrisk'] = currentrisk
        evading.input['asteroid_time'] = asteroid_t
        evading.input['mine_distance'] = mine_distance
        evading.compute()

        safetystatus = evading.output['ship_evade']

        #target or move based on safety status
        if safetystatus < 0:
            shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
            shooting.input['bullet_time'] = bullet_t
            shooting.input['theta_delta'] = shooting_theta
            shooting.compute()
            outputs = shooting.output
            raw_thrust = 0

        elif safetystatus >= 0:
            movement = ctrl.ControlSystemSimulation(self.movement_control,flush_after_run=1)
            movement.input['currentrisk'] = currentrisk
            movement.input['bestdirection'] = bestdirection
            movement.compute()
            outputs = movement.output
            raw_thrust = float(outputs.get('ship_thrust', 0.0))
        
        

        turn_rate = float(outputs.get('ship_turn', 0.0))

        #test if we should fire
        firesim.input['shootasteroid'] = shootasteroid
        firesim.compute()
        fireoutput = firesim.output

        fire_value = fireoutput.get('ship_fire', -1.0)
        fire = bool(fire_value >= 0)

        raw_thrust *= 0.3
        MAX_THRUST = 1000.0
        thrust = max(-MAX_THRUST, min(MAX_THRUST, raw_thrust))

        #help ship come to a stop since the fuzzy logic doesn't really consider drifting too well
        # if (thrust < 50 and thrust > -50 and abs(ship_state['speed']) > 10):
        #     if ship_state['speed'] < 0:
        #         thrust = -1000
        #     else:
        #         thrust = 1000


        
        # If asteroid is approaching soon
        if asteroid_t < 2.0:
            # And it's mostly behind us (more than 90 degrees off the nose)
            if abs(asteroid_theta) > math.pi / 2:
                # We want to move forward in this case
                thrust = -abs(thrust) if thrust != 0 else -150.0

        # If no asteroid is near barely move
        if asteroid_t > 5.0:
            thrust *= 0.2
        
        mine_value = outputs.get('ship_mine', -1.0)
        drop_mine = bool(mine_value >= 0)

        """
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
        """

        #DEBUG
        # print(f"safetystatus: {safetystatus}, bestdirection: {bestdirection}, currentrisk: {currentrisk}")
        #print(f"thrust: {thrust}, currentrisk: {currentrisk}, bestdirection: {bestdirection}, heading: {ship_state['heading']}")
        #print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        # THe games thrust sign is opposite of our convention so keep the - in front of thrust 
        return -thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Controller"
    
if __name__ == '__main__':

    print("Starting Genetic Algorithm Optimization...")

    # --- GA gene / chromosome generation ---

    def generate_random_gene():
        # Single MF gene in [0,1]
        return random.uniform(0.0, 1.0)

    def generate_random_chromosome():
        # Return a Python list of 12 floats
        return [generate_random_gene() for _ in range(NUM_GENES)]

    # --- Scenario for GA training ---

    ga_training_scenario = Scenario(
        name='GA Training Scenario',
        num_asteroids=10,
        ship_states=[
            {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
        ],
        map_size=(1000, 800),
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

    # Use TrainerEnvironment for speed (no graphics)
    ga_game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 2,
                 'graphics_obj': None,
                 'frequency': 30}

    # --- Fitness function ---

    def kessler_fitness(chromosome):
        # Convert Gene objects to list of floats
        runs_per_chromosome = 4
        total_cost = 0.0
        mf_list = [gene.value for gene in chromosome]


   
        for _ in range(runs_per_chromosome):
            env = TrainerEnvironment(settings=ga_game_settings)
            ctrlr = controller(mf_chromosome=mf_list)

            pre = time.perf_counter()
            score, perf = env.run(scenario=ga_training_scenario, controllers=[ctrlr])
            print(f"Finished game #{game_num[0]}")
            game_num[0] += 1
            alive_time = time.perf_counter()-pre
            # print('Scenario eval time: '+str(alive_time))
            team = score.teams[0]

            deaths = team.deaths
            asteroids_hit = team.asteroids_hit

            # Minimize
            cost = (
                asteroids_hit
            )
            total_cost += cost
        return total_cost / runs_per_chromosome

    ga = EasyGA.GA()
    ga.chromosome_length = NUM_GENES
    ga.population_size = 2       # can raise for overnight
    ga.max_generations  = 1      # can also raise
    ga.generation_goal = 2
    ga.target_fitness_type = 'max'

    # Either use chromosome_impl OR gene_impl; pick ONE pattern.
    # Simpler: use gene_impl + chromosome_length
    ga.gene_impl = generate_random_gene
    # (You can delete ga.chromosome_impl if set earlier.)
    ga.fitness_function_impl = kessler_fitness

    print("Starting GA to tune MFs...")
    start = time.time()
    ga.evolve()
    elapsed = time.time() - start
    print(f"GA finished in {elapsed:.1f} seconds")

    ga.sort_by_best_fitness()
    print("\n=== TOP 10 CHROMOSOMES (HIGHEST COST FIRST) ===")
    for rank, chrom in enumerate(ga.population[:20], start=1):
        genes = [gene.value for gene in chrom]
        print(f"\nRank {rank}: Fitness (cost) = {chrom.fitness:.3f}")
        print("Genes:", genes)


    best = ga.population[0]
    best_genes = [g.value for g in best]
    print("\n=== BEST CHROMOSOME OVERALL ===")
    print("Best fitness (cost):", best.fitness)
    print("Best genes:", best_genes)