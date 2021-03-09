const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.2;
const LAMBDA = 0.01;

class Orchestrator {
    
    constructor(maxStepsPerGame, model, memory, discountRate) {
        this.maxStepsPerGame = maxStepsPerGame;
        this.model = model;
        this.memory = memory;
        this.eps = MAX_EPSILON;
        this.steps = 0;
        this.discountRate = discountRate;
        
        this.rewardStore = new Array();
        this.maxPositionStore = new Array();
    }
    
    
    async run() {
        // reset the simulation. And get a new scene_data to create env.
        console.log("resetting sim");
        await api.doReset();
        api.latest_scene_data = null;
        console.log("Getting scene data");
        let scene_data = await api.getSceneData();
        let env = new Environment(scene_data);
        console.log("Initial state tensor");
        let state = env.getStateTensor();
        let totalReward = 0;
        let step = 0;
        
        console.log("Beginning game, max steps =  " + this.maxStepsPerGame);
        while (step < this.maxStepsPerGame) {
            console.log("Beginning step " + step);
            const action = this.model.chooseAction(state, this.eps)
            console.log("step " + step + ": action = " + action);
            const done = await env.update(action);
            const reward = done? -1 : 1;
            console.log("Executed action done = " + done);

            
            let nextState = env.getStateTensor();
            
            if (done) nextState = null;
            
            this.memory.addSample([state, action, reward, nextState]);
            
            this.steps += 1;
            this.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps);
            
            state = nextState;
            totalReward += reward;
            step += 1;
            
            if (done || step == this.maxStepsPerGame) {
                this.rewardStore.push(totalReward);
                break;
            }
        }
        await this.replay();
        
        return totalReward;
    }
    
    async replay() {
        console.log('replaying');
        const batch = this.memory.sample(this.model.batchSize);
        const states = batch.map(([state, , , ]) => state);
        const nextStates = batch.map(
            ([, , , nextState]) => nextState ? nextState : tf.zeros([1, this.model.numStates])
        );
        
        // print States
        console.log(states.length)
        console.log(nextStates.length)

        console.log("...with states = ");
        states.forEach(s => s.print());
        // print Next States
        console.log("...with next states = ");
        nextStates.forEach(s => s.print());
        
        
        // Predict the values of each action at each state
        const qsa = states.map((state) => this.model.predict(state));

        
        // Predict the values of each action at each next state
        const qsad = nextStates.map((nextState) => this.model.predict(nextState));

        let x = new Array();
        let y = new Array();
        
        // Update the states rewards with the discounted next states rewards
        console.log("updating rewards");
        batch.forEach(
            ([state, action, reward, nextState], index) => {
                console.log(`qsa[${index}] = `);
                qsa[index].print();
                console.log(`qsad[${index}] = `);
                qsad[index].print();
                
                const currentQ = qsa[index].dataSync();
                let action_index = action + 1;
                
                console.log(`action = ${action}, action_i = ${action_index}, reward = ${reward}`);
                console.log(`currentQ = ${currentQ.toString()}`);
                
                currentQ[action_index] = nextState ? reward + this.discountRate * qsad[index].max().dataSync() : reward;
                
                console.log(`new currentQ = ${currentQ.toString()}`);

                
                x.push(state.dataSync());
                y.push(currentQ);
            }
        );
        
        qsa.forEach((state) => state.dispose());
        qsad.forEach((state) => state.dispose());
        
        x = tf.tensor2d(x, [x.length, this.model.numStates]);
        y = tf.tensor2d(y, [y.length, this.model.numActions]);
        
        await this.model.train(x, y);
        
        x.dispose();
        y.dispose();
    }
}