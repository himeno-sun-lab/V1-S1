#!/usr/local/bin/python

import os
import subprocess
import time

class WBNao:

    remotehost = 'dcprg.numericalbrain.org'
    remotedir = '/tmp'
    direct_file = 'd.txt'
    popvec_file = 'p.txt'
    reward_file = 'r.txt'

    rm = 'rm'
    scp = 'scp'
    ssh = 'ssh'

    def putfile ( self, filename, content ):
        f = open(filename, 'w')
        f.write(content)
        f.close()
        dst = self.remotehost + ':' + self.remotedir
        subprocess.call([self.scp, filename, dst ])

    def getfile ( self, filename ):
        src = self.remotehost + ':' + self.remotedir + '/' + filename
        subprocess.call([self.scp, src, '.'])
        if os.path.exists(filename):
            f = open(filename, 'r')
            content = f.read()
            f.close()
            return content
        else:
            return None

    def putDesiredDirection( self, direction ):
        self.putfile(self.direct_file, direction)

    def getDesiredDirection(self):
        while True:
            s = self.getfile(self.direct_file)
            if s is None:
                print("Waiting")
                time.sleep(10)
            else:
                return s

    def putPopulationVector(self, popvec):
        s = ','.join(map(str, popvec)) 
        self.putfile(self.popvec_file, s)

    def getPopulationVector(self):
        while True:
            s = self.getfile(self.popvec_file).split(',')
            if s is None:
                print("Waiting")
                time.sleep(10)
            else:
                return [int(str) for str in s]

#    def putReward(self, reward):
        # To be implemented

#    def getReward(self):
        # To be implemented

    def clean(self):
        if os.path.exists(self.direct_file):
            os.remove(self.direct_file)
        if os.path.exists(self.popvec_file):
            os.remove(self.popvec_file)

    def init_server(self):
        f1 = self.remotedir + '/' + self.direct_file
        f2 = self.remotedir + '/' + self.popvec_file
        subprocess.call([self.ssh, self.remotehost, self.rm, f1, f2])

def main():

    wbnao = WBNao()

    wbnao.clean()
    wbnao.init_server() # must be invoked only from NAO

    # ... Here, NAO reads the computer sceeen and determines the target direction ...

    wbnao.putDesiredDirection('L') # NAO tells WB the target direction, one of 'L', 'C', 'R'
    print (wbnao.getDesiredDirection()) # WB receives the target direction 

    # ... Here, WB calculates the vector of population activity in M1 neurons ...

    wbnao.putPopulationVector( [10, 20, 10, 0, 0, 0 ] ) # WB sends the population vector
    print (wbnao.getPopulationVector()) # NAO receives the population vector

    # ... Here, the population vector is transferred to the joint angle vectors ...
    # ... Here, NAO moves the arm according to the joint angle vectors ...

    # ... Here, NAO calculates the reward, and send the value to WB ...
    # ... Here, WB undergo RL based on the given RL ...

    # ... Repeat ...

if __name__ == "__main__":
    main()
