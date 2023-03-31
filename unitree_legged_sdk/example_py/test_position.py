import sys
import time
import math

# sys.path.append('../lib/python/amd64')
import robot_interface as sdk

def subs(L1, L2):
    l = []
    for l1, l2 in zip(L1, L2):
        l.append(l1-l2)
    return l


if __name__ == '__main__':

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    # udp = sdk.UDP(HIGHLEVEL, 8090, "192.168.123.161", 8082)
    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.12.1", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    for _ in range(10):
        cmd.mode = 0
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(0.1)

    time.sleep(1.0)

    udp.Recv()
    udp.GetRecv(state)
    start_position = state.position

    print('Start: ', state.position)

    
    cmd.mode = 2
    cmd.gaitType = 1
    cmd.velocity = [0.2, 0.] # -1  ~ +1
    cmd.yawSpeed = 0.
    cmd.bodyHeight = 0.

    for i in range(5):
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(1.0)
        udp.Recv()
        udp.GetRecv(state)
        print('Current: ', subs(state.position, start_position))

    for i in range(5):
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(1.0)
        udp.Recv()
        udp.GetRecv(state)
        print('Current: ', subs(state.position, start_position))

    time.sleep(1.)
    udp.Recv()
    udp.GetRecv(state)
    print('End: ', subs(state.position, start_position))
    print()