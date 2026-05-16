from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.log import setLogLevel
import time

def topology():
        
    net = Mininet(controller=None)
    g0_s0=net.addSwitch('g0_s0')
    g0_s1=net.addSwitch('g0_s1')
    g1_s0=net.addSwitch('g1_s0')
    g1_s1=net.addSwitch('g1_s1')
    g2_s0=net.addSwitch('g2_s0')
    g2_s1=net.addSwitch('g2_s1')
    g0_s0_h1 = net.addHost('g0_s0_h1')
    g0_s1_h1 = net.addHost('g0_s1_h1')
    g1_s0_h1 = net.addHost('g1_s0_h1')
    g1_s1_h1 = net.addHost('g1_s1_h1')
    g2_s0_h1 = net.addHost('g2_s0_h1')
    g2_s1_h1 = net.addHost('g2_s1_h1')
    net.addLink(g0_s0_h1, g0_s0)
    net.addLink(g0_s1_h1, g0_s1)
    net.addLink(g1_s0_h1, g1_s0)
    net.addLink(g1_s1_h1, g1_s1)
    net.addLink(g2_s0_h1, g2_s0)
    net.addLink(g2_s1_h1, g2_s1)
    net.addLink(g0_s0, g0_s1)
    net.addLink(g1_s0, g1_s1)
    net.addLink(g2_s0, g2_s1)
    net.addLink(g0_s0, g1_s0) 
    net.addLink(g1_s0, g2_s0)
    net.addLink(g0_s0, g2_s0)
    net.start()
    
    print("\n[!] Configuring Dragonfly Switches...")

    g0_s0.cmd(f'ovs-vsctl set-fail-mode g0_s0 standalone')
    g0_s1.cmd(f'ovs-vsctl set-fail-mode g0_s1 standalone')
    g1_s0.cmd(f'ovs-vsctl set-fail-mode g1_s0 standalone')
    g1_s1.cmd(f'ovs-vsctl set-fail-mode g1_s1 standalone')
    g2_s0.cmd(f'ovs-vsctl set-fail-mode g2_s0 standalone')
    g2_s1.cmd(f'ovs-vsctl set-fail-mode g2_s1 standalone')

    g0_s0.cmd(f'ovs-vsctl set Bridge g0_s0 stp_enable=true')
    g0_s1.cmd(f'ovs-vsctl set Bridge g0_s1 stp_enable=true')
    g1_s0.cmd(f'ovs-vsctl set Bridge g1_s0 stp_enable=true')
    g1_s1.cmd(f'ovs-vsctl set Bridge g1_s1 stp_enable=true')
    g2_s0.cmd(f'ovs-vsctl set Bridge g2_s0 stp_enable=true')
    g2_s1.cmd(f'ovs-vsctl set Bridge g2_s1 stp_enable=true')
	
    print("\n Please wait 30 seconds :")
    time.sleep(30)
    print("\nNetwork Established ! Go ahead.\n")

    CLI(net)
    net.stop()
if __name__ == '__main__':
    setLogLevel('info')
    topology()
