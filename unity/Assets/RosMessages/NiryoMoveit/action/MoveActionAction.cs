using System.Collections.Generic;
using RosMessageGeneration;


namespace RosMessageTypes.NiryoMoveit
{
    public class MoveActionAction : Action<MoveActionActionGoal, MoveActionActionResult, MoveActionActionFeedback, MoveActionGoal, MoveActionResult, MoveActionFeedback>
    {
        public const string RosMessageName = "niryo_moveit/MoveActionAction";

        public MoveActionAction() : base()
        {
            this.action_goal = new MoveActionActionGoal();
            this.action_result = new MoveActionActionResult();
            this.action_feedback = new MoveActionActionFeedback();
        }

        public override List<byte[]> SerializationStatements()
        {
            var listOfSerializations = new List<byte[]>();
            listOfSerializations.AddRange(this.action_goal.SerializationStatements());
            listOfSerializations.AddRange(this.action_result.SerializationStatements());
            listOfSerializations.AddRange(this.action_feedback.SerializationStatements());

            return listOfSerializations;
        }

        public override int Deserialize(byte[] data, int offset)
        {
            offset = this.action_goal.Deserialize(data, offset);
            offset = this.action_result.Deserialize(data, offset);
            offset = this.action_feedback.Deserialize(data, offset);

            return offset;
        }

    }
}
