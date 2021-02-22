
class SubscriberQueue {
    constructor(name, subscribe_call, max_size=1) {
        this.name = name;
        this.queue = new Array();
        
        // Bound size of message queue.
        this.queue.push = function (){
            if (this.length >= max_size) {
                this.shift();
            }
            return Array.prototype.push.apply(this,arguments);
        }

        subscribe_call.on('data', this.onData.bind(this));
        subscribe_call.on('end', this.onEnd.bind(this));
        subscribe_call.on('error', this.onError.bind(this));
        subscribe_call.on('status', this.onStatus.bind(this));
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