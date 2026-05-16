from mininet.net import Mininet
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.node import OVSSwitch
from mininet.log import setLogLevel
import time

class Star(Topo):
    def build(self, n=4):
        switch = self.addSwitch('s1', failMode='standalone')
        for i in range(1, n + 1):
            host = self.addHost(f'h{i}')
            self.addLink(host, switch)

def run_star():
    topo = Star(n=4)
    net = Mininet(topo=topo, switch=OVSSwitch)
    
    net.start()

    print("\nPlease wait 10 seconds")
    time.sleep(10)
    print("Network Established. Go ahead!")

    CLI(net)

    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run_star()