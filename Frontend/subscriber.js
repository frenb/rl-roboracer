var messages = require('./proto/virtual_endpoint/proto/ros_service_pb');

class SubscriberQueue {
    constructor(name, msg_type, client, max_size=1) {
        this.name = name;
        this.queue = new Array();
        
        // Bound size of message queue.
        this.queue.push = function (){
            if (this.length >= max_size) {
                this.shift();
            }
            return Array.prototype.push.apply(this,arguments);
        }

        let request = new messages.SubscribeRequest();
        request.setTopic(name);
        request.setMsgType(msg_type);
        this.call = client.subscribe(request);

        this.call.on('data', this.onData.bind(this));
        this.call.on('end', this.onEnd.bind(this));
        this.call.on('error', this.onError.bind(this));
        this.call.on('status', this.onStatus.bind(this));
    }

    peek() {
        return this.queue[this.queue.length-1];
    }

    onData(topic_message) {
        try {
            this.queue.push({data: JSON.parse(topic_message.getData()), timestamp: Date.now()});
          } catch (e) {
            console.log(e);
        }
    }

    onEnd() {
        console.log(this.name + " - end");

    }

    onError(e) {
        console.log(this.name + " - error: " + e);

    }

    onStatus(status) {
        console.log(this.name + " - status: " + status);
    }
}

module.exports = SubscriberQueue