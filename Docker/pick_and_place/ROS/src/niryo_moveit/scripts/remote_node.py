#!/usr/bin/env python

from __future__ import print_function

import rospy
import asyncio

from virtual_endpoint import VirtualNode

def remote_node():
    rospy.init_node('remote_node', anonymous=True)
    rospy.loginfo(rospy.get_caller_id() + " I Remote Node Started")
    v_node = VirtualNode()
    asyncio.run(v_node.main())

if __name__ == "__main__":
    remote_node()