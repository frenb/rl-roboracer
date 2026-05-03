using System.Collections.Generic;
using RosMessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Actionlib;

namespace RosMessageTypes.NiryoMoveit
{
    public class MoveActionActionGoal : ActionGoal<MoveActionGoal>
    {
        public const string RosMessageName = "niryo_moveit/MoveActionActionGoal";

        public MoveActionActionGoal() : base()
        {
            this.goal = new MoveActionGoal();
        }

        public MoveActionActionGoal(Header header, GoalID goal_id, MoveActionGoal goal) : base(header, goal_id)
        {
            this.goal = goal;
        }
        public override List<byte[]> SerializationStatements()
        {
            var listOfSerializations = new List<byte[]>();
            listOfSerializations.AddRange(this.header.SerializationStatements());
            listOfSerializations.AddRange(this.goal_id.SerializationStatements());
            listOfSerializations.AddRange(this.goal.SerializationStatements());

            return listOfSerializations;
        }

        public override int Deserialize(byte[] data, int offset)
        {
            offset = this.header.Deserialize(data, offset);
            offset = this.goal_id.Deserialize(data, offset);
            offset = this.goal.Deserialize(data, offset);

            return offset;
        }

    }
}
