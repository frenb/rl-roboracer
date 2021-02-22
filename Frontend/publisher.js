var messages = require('./proto/virtual_endpoint/proto/ros_service_pb');

class Publisher {
    constructor(topic, msg_type, client) {
        this.topic = topic;
        this.msg_type = msg_type;
        this.client = client;
    }

    publish(data) {
        let request = new messages.PublishRequest();
        request.setTopic(this.topic);
        request.setMsgType(this.msg_type);
        request.setData(JSON.stringify(data));
        console.log(this.topic + " publishing: " + data);
        this.client.publish(request, function(err, response_unused) {
            console.log(this.topic + " error publishing " + err);
        });
    }
}

module.exports = Publisher