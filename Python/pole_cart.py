import asyncio
import random

from api import RobotApi


async def start():
    api = RobotApi()
    await api.Initialize()
    await api.DoReset()
    await random_walk(api)


async def random_walk(api):
    while True:
        scene_data = await api.GetSceneData()
        positions = {
            'joint_00': scene_data['joint_00'] + random.uniform(-1, 1) * 3.14 / 180.0,
            'joint_01': scene_data['joint_01'] + random.uniform(-1, 1) * 3.14 / 180.0,
            'joint_02': scene_data['joint_02'] + random.uniform(-1, 1) * 3.14 / 180.0,
            'joint_03': scene_data['joint_03'] + random.uniform(-1, 1) * 3.14 / 180.0,
            'joint_04': scene_data['joint_04'] + random.uniform(-1, 1) * 3.14 / 180.0,
            'joint_05': scene_data['joint_05'] + random.uniform(-1, 1) * 3.14 / 180.0,
        }
        action = {
            'cmd_type': 4,
            'positions': positions
        }
        await api.DoMove({'cmd': action})

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start())
    loop.close()




