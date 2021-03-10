
const NUM_ACTIONS = 7;

const INDEX_TO_ACTION = {
    0: -2,
    1: -1,
    2: -0.5,
    3: 0,
    4: 0.5,
    5: 1,
    6: 2
};

const ACTION_TO_INDEX = {
    "-2": 0,
    "-1": 1,
    "-0.5": 2,
    0: 3,
    0.5: 4,
    1: 5,
    2: 6
};

function indexToAction(index) {
    return INDEX_TO_ACTION[index];
}

function actionToIndex(action) {
    return ACTION_TO_INDEX[action];
}


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
        return tf.tidy(() => this.network.predict(states));
    }
    
    async train(xBatch, yBatch) {
        await this.network.fit(xBatch, yBatch);
    }
    
    // Returns action -1, 0, 1
    chooseAction(state, eps) {
        if (Math.random() < eps) {
            return indexToAction(Math.floor(Math.random() * this.numActions));
        } else {
            return tf.tidy(() => {
                return indexToAction(this.network.predict(state).argMax(1).dataSync()[0]);
            });
        }
    }
}