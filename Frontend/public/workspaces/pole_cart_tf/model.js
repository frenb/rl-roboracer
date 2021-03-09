class Model {
    constructor(hiddenLayerSizesOrModel, numStates, numActions, batchSize) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.batchSize = batchSize;
        
        if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
            this.network = hiddenLayerSizesOrModel;
            this.network.summary();
            this.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
        } else {
            this.defineModel(hiddenLayerSizesOrModel);
        }
    }
    
    defineModel(hiddenLayerSizes) {
        if (!Array.isArray(hiddenLayerSizes)) {
            hiddenLayerSizes = [hiddenLayerSizes];
        }
        this.network = tf.sequential();
        hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
            this.network.add(tf.layers.dense({
                units: hiddenLayerSize,
                activiation: 'relu',
                inputShape: i == 0 ? [this.numStates] : undefined
            }));
        });
        this.network.add(tf.layers.dense({units: this.numActions}));
        
        this.network.summary();
        this.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    }
    
    predict(states) {
        console.log("model.predict: ");
        states.print();
        let ret =  tf.tidy(() => this.network.predict(states));
        console.log("returning: ");
        ret.print();
        return ret;
    }
    
    async train(xBatch, yBatch) {
        await this.network.fit(xBatch, yBatch);
    }
    
    // Returns action -1, 0, 1
    chooseAction(state, eps) {
        if (Math.random() < eps) {
            return Math.floor(Math.random() * this.numActions) - 1;
        } else {
            return tf.tidy(() => {
                const logits = this.network.predict(state);
                const sigmoid = tf.sigmoid(logits);
                const probs = tf.div(sigmoid, tf.sum(sigmoid));
                return tf.multinomial(probs, 1).dataSync()[0] - 1;
            });
        }
    }
}