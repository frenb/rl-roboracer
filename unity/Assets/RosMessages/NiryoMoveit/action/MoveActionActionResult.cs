using System.Collections.Generic;
using RosMessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Actionlib;

namespace RosMessageTypes.NiryoMoveit
{
    public class MoveActionActionResult : ActionResult<MoveActionResult>
    {
        public const string RosMessageName = "niryo_moveit/MoveActionActionResult";

        public MoveActionActionResult() : base()
        {
            this.result = new MoveActionResult();
        }

        public MoveActionActionResult(Header header, GoalStatus status, MoveActionResult result) : base(header, status)
        {
            this.result = result;
        }
        public override List<byte[]> SerializationStatements()
        {
            var listOfSerializations = new List<byte[]>();
            listOfSerializations.AddRange(this.header.SerializationStatements());
            listOfSerializations.AddRange(this.status.SerializationStatements());
            listOfSerializations.AddRange(this.result.SerializationStatements());

            return listOfSerializations;
        }

        public override int Deserialize(byte[] data, int offset)
        {
            offset = this.header.Deserialize(data, offset);
            offset = this.status.Deserialize(data, offset);
            offset = this.result.Deserialize(data, offset);

            return offset;
        }

    }
}
