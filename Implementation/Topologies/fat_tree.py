from mininet.topo import Topo
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel
import time


def create_fat_tree_topology(k=4):

    topo = Topo()

    # Number of pods
    num_pods = k

    # Number of core switches
    num_core_switches = (k // 2) ** 2

    # Store core switches
    cores = []

    # Create core switches
    for i in range(num_core_switches):
        core = topo.addSwitch(f'c{i+1}')
        cores.append(core)

    # Create pods
    for p in range(num_pods):

        aggs = []
        edges = []

        # Create aggregation and edge switches
        for i in range(k // 2):

            agg = topo.addSwitch(f'p{p}_a{i+1}')
            edge = topo.addSwitch(f'p{p}_e{i+1}')

            aggs.append(agg)
            edges.append(edge)

        # Connect aggregation switches to core switches
        for agg in aggs:
            for core in cores:
                topo.addLink(agg, core)

        # Connect edge switches to aggregation switches
        for edge in edges:
            for agg in aggs:
                topo.addLink(edge, agg)

        # Add hosts to edge switches
        for i, edge in enumerate(edges):

            for h_idx in range(k // 2):

                host = topo.addHost(f'p{p}_e{i+1}_h{h_idx+1}')

                topo.addLink(host, edge)

    return topo


def run():

    topo = create_fat_tree_topology(k=4)
    net = Mininet(
        topo=topo,
        controller=None
    )

    net.start()

    # Enable standalone mode and STP
    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set-fail-mode {sw.name} standalone')
        sw.cmd(f'ovs-vsctl set bridge {sw.name} stp_enable=true')

    print("Please wait 30 seconds: ")
    time.sleep(30)
    print("\nNetwork Established. Go ahead! \n")
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run()