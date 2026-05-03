using System.Collections.Generic;
using RosMessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Actionlib;

namespace RosMessageTypes.NiryoMoveit
{
    public class MoveActionActionFeedback : ActionFeedback<MoveActionFeedback>
    {
        public const string RosMessageName = "niryo_moveit/MoveActionActionFeedback";

        public MoveActionActionFeedback() : base()
        {
            this.feedback = new MoveActionFeedback();
        }

        public MoveActionActionFeedback(Header header, GoalStatus status, MoveActionFeedback feedback) : base(header, status)
        {
            this.feedback = feedback;
        }
        public override List<byte[]> SerializationStatements()
        {
            var listOfSerializations = new List<byte[]>();
            listOfSerializations.AddRange(this.header.SerializationStatements());
            listOfSerializations.AddRange(this.status.SerializationStatements());
            listOfSerializations.AddRange(this.feedback.SerializationStatements());

            return listOfSerializations;
        }

        public override int Deserialize(byte[] data, int offset)
        {
            offset = this.header.Deserialize(data, offset);
            offset = this.status.Deserialize(data, offset);
            offset = this.feedback.Deserialize(data, offset);

            return offset;
        }

    }
}
