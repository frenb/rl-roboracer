import asyncio

from api import RobotApi

# Quaternion peripindicular to the niryo table.
TABLE_PERPINDICULAR = {'x' : -0.5, 'y' : -0.5, 'z' : 0.5, 'w' : -0.5}

async def main():
    api = RobotApi()
    await api.Initialize()

    scene_data = await api.GetSceneData()

    print('Executing pick trajactory...')
    posePoint = scene_data['object_location']
    posePoint['z'] = posePoint['z'] + 0.10 # stop just above object.
    posePoint['y'] = posePoint['y'] - 0.01
    posePoint['x'] = posePoint['x'] - 0.01 
    goal = {'position': posePoint, 'orientation': TABLE_PERPINDICULAR}
    plan = await api.GetPlan(goal)
    await api.DoTrajectory(plan['trajectory'])
    print('---> Done')

    print("Opening gripper...")
    await api.DoOpenGripper()
    print('---> Done')
    
    print('Executing lower trajectory...')
    posePoint['z'] = posePoint['z'] - 0.04
    plan = await api.GetPlan(goal)
    await api.DoTrajectory(plan['trajectory'])
    print('---> Done')
    
    print("Closing gripper...")
    await api.DoCloseGripper()
    print('---> Done')
    
    print('Executing place trajectory...')
    posePoint = scene_data['target_location']
    posePoint['z'] = posePoint['z'] + 0.15 # Just above.
    goal = {'position': posePoint, 'orientation': TABLE_PERPINDICULAR}
    plan = await api.GetPlan(goal)
    await api.DoTrajectory(plan['trajectory'])
    print('---> Done')
     
    print("Opening gripper...")
    await api.DoOpenGripper()
    print('---> Done')


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())